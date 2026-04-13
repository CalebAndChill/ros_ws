#!/usr/bin/env python3
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    from sensor_msgs import point_cloud2

from tf2_ros import Buffer, TransformListener, TransformException

from yolo_msgs.msg import DetectionArray


class BestMaskMainMassPCA(Node):
    def __init__(self):
        super().__init__('best_mask_main_mass_pca')

        # ---------------- Parameters ----------------
        self.declare_parameter('detections_topic', '/yolo/detections_3d')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')

        self.declare_parameter('cloud_topic', '/best_mask_cloud')
        self.declare_parameter('marker_topic', '/best_mask_pca_markers')

        self.declare_parameter('depth_units_divisor', 1000.0)   # 16UC1 mm -> m
        self.declare_parameter('sample_stride', 4)
        self.declare_parameter('mask_shrink_px', 0)
        self.declare_parameter('min_score', 0.0)
        self.declare_parameter('allowed_classes', [])
        self.declare_parameter('max_points', 40000)
        self.declare_parameter('z_min', 0.05)
        self.declare_parameter('z_max', 10.0)
        self.declare_parameter('use_roi', True)
        self.declare_parameter('roi_expand_px', 0)

        # Main-mass filtering
        self.declare_parameter('use_main_component', True)
        self.declare_parameter('min_component_pixels', 20)
        self.declare_parameter('depth_band_m', 0.08)

        # "" = keep native depth/camera frame
        # set to torso_link later if TF is available
        self.declare_parameter('output_frame', '')

        self.det_topic = self.get_parameter('detections_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.cloud_topic = self.get_parameter('cloud_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value

        self.depth_units_divisor = float(self.get_parameter('depth_units_divisor').value)
        self.sample_stride = int(self.get_parameter('sample_stride').value)
        self.mask_shrink_px = int(self.get_parameter('mask_shrink_px').value)
        self.min_score = float(self.get_parameter('min_score').value)
        self.allowed_classes = set(self.get_parameter('allowed_classes').value or [])
        self.max_points = int(self.get_parameter('max_points').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.use_roi = bool(self.get_parameter('use_roi').value)
        self.roi_expand_px = int(self.get_parameter('roi_expand_px').value)

        self.use_main_component = bool(self.get_parameter('use_main_component').value)
        self.min_component_pixels = int(self.get_parameter('min_component_pixels').value)
        self.depth_band_m = float(self.get_parameter('depth_band_m').value)

        self.output_frame = str(self.get_parameter('output_frame').value).strip()
        self.add_on_set_parameters_callback(self._on_param_change)

        # ---------------- State ----------------
        self.bridge = CvBridge()
        self.depth = None
        self.depth_frame = 'camera_color_optical_frame'
        self.fx = self.fy = self.cx = self.cy = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- QoS ----------------
        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )

        # ---------------- Subs / pubs ----------------
        self.sub_info = self.create_subscription(
            CameraInfo, self.info_topic, self.cb_info, 10
        )
        self.sub_depth = self.create_subscription(
            Image, self.depth_topic, self.cb_depth, depth_qos
        )
        self.sub_det = self.create_subscription(
            DetectionArray, self.det_topic, self.cb_det, 10
        )

        self.pub_cloud = self.create_publisher(PointCloud2, self.cloud_topic, 1)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 1)

        self.get_logger().info(
            f'BestMaskMainMassPCA running.\n'
            f'  detections_topic: {self.det_topic}\n'
            f'  depth_topic:      {self.depth_topic}\n'
            f'  camera_info_topic:{self.info_topic}\n'
            f'  cloud_topic:      {self.cloud_topic}\n'
            f'  marker_topic:     {self.marker_topic}\n'
            f'  output_frame:     {self.output_frame if self.output_frame else "[depth frame]"}\n'
            f'  use_main_component:{self.use_main_component}\n'
            f'  min_component_pixels:{self.min_component_pixels}\n'
            f'  depth_band_m:     {self.depth_band_m}'
        )

    # --------------------------------------------------
    # Parameter callback
    # --------------------------------------------------
    def _on_param_change(self, params):
        for p in params:
            if p.name == 'output_frame':
                self.output_frame = str(p.value).strip()
                self.get_logger().info(
                    f'Updated output_frame -> {self.output_frame if self.output_frame else "[depth frame]"}'
                )
            elif p.name == 'depth_band_m':
                self.depth_band_m = float(p.value)
                self.get_logger().info(f'Updated depth_band_m -> {self.depth_band_m}')
            elif p.name == 'min_component_pixels':
                self.min_component_pixels = int(p.value)
                self.get_logger().info(f'Updated min_component_pixels -> {self.min_component_pixels}')
            elif p.name == 'use_main_component':
                self.use_main_component = bool(p.value)
                self.get_logger().info(f'Updated use_main_component -> {self.use_main_component}')

        return SetParametersResult(successful=True)

    # --------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------
    def cb_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def cb_depth(self, msg: Image):
        self.depth_frame = msg.header.frame_id if msg.header.frame_id else self.depth_frame
        self.depth = self.bridge.imgmsg_to_cv2(msg)

    def cb_det(self, det_arr: DetectionArray):
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.depth is None or self.fx is None:
            self.pub_markers.publish(marker_array)
            return

        best_det = self._pick_best_detection(det_arr)
        if best_det is None:
            self.pub_markers.publish(marker_array)
            return

        pts = self._points_from_detection(best_det)
        if pts is None or pts.shape[0] < 10:
            self.pub_markers.publish(marker_array)
            return

        frame_id = self.depth_frame

        # Optional transform to another frame, e.g. torso_link
        if self.output_frame and self.output_frame != self.depth_frame:
            transformed = self._transform_points(pts, self.depth_frame, self.output_frame)
            if transformed is None:
                self.get_logger().warn(
                    f'Could not transform points from {self.depth_frame} to {self.output_frame}'
                )
                self.pub_markers.publish(marker_array)
                return
            pts = transformed
            frame_id = self.output_frame

        pca = self._compute_pca_obb(pts)
        if pca is None:
            self.pub_markers.publish(marker_array)
            return

        centroid, eigvecs, extents, corners = pca

        # Cap published cloud size only
        cloud_pts = pts
        if self.max_points > 0 and cloud_pts.shape[0] > self.max_points:
            sel = np.random.choice(cloud_pts.shape[0], self.max_points, replace=False)
            cloud_pts = cloud_pts[sel]

        stamp = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        cloud_msg = point_cloud2.create_cloud_xyz32(header, cloud_pts.tolist())
        self.pub_cloud.publish(cloud_msg)

        marker_array.markers.extend(
            self._build_markers(
                frame_id=frame_id,
                stamp=stamp,
                det=best_det,
                centroid=centroid,
                eigvecs=eigvecs,
                extents=extents,
                corners=corners,
            )
        )
        self.pub_markers.publish(marker_array)

    # --------------------------------------------------
    # Detection selection
    # --------------------------------------------------
    def _pick_best_detection(self, det_arr: DetectionArray):
        best_det = None
        best_score = -math.inf

        for det in det_arr.detections:
            if det.score < self.min_score:
                continue
            if self.allowed_classes and det.class_name not in self.allowed_classes:
                continue

            has_mask = hasattr(det, 'mask') and det.mask and len(det.mask.data) >= 3
            has_bbox = hasattr(det, 'bbox')

            if not has_mask and not has_bbox:
                continue

            if det.score > best_score:
                best_score = det.score
                best_det = det

        return best_det

    # --------------------------------------------------
    # Point cloud generation from 2D mask / bbox
    # --------------------------------------------------
    def _points_from_detection(self, det):
        depth = self.depth
        H, W = depth.shape[:2]

        # ROI in full-res image coordinates
        if hasattr(det, 'mask') and det.mask and len(det.mask.data) >= 3:
            poly = np.array([(p.x, p.y) for p in det.mask.data], dtype=np.float32)
            x1 = int(np.floor(np.min(poly[:, 0])))
            y1 = int(np.floor(np.min(poly[:, 1])))
            x2 = int(np.ceil(np.max(poly[:, 0])))
            y2 = int(np.ceil(np.max(poly[:, 1])))
        else:
            cx2d = int(det.bbox.center.position.x)
            cy2d = int(det.bbox.center.position.y)
            bw = int(det.bbox.size.x)
            bh = int(det.bbox.size.y)
            x1 = cx2d - bw // 2
            x2 = cx2d + (bw - bw // 2)
            y1 = cy2d - bh // 2
            y2 = cy2d + (bh - bh // 2)
            poly = None

        if self.use_roi:
            x1 = max(0, x1 - self.roi_expand_px)
            y1 = max(0, y1 - self.roi_expand_px)
            x2 = min(W, x2 + self.roi_expand_px)
            y2 = min(H, y2 + self.roi_expand_px)
        else:
            x1, y1, x2, y2 = 0, 0, W, H

        if x2 <= x1 or y2 <= y1:
            return None

        xs = np.arange(x1, x2, self.sample_stride, dtype=np.int32)
        ys = np.arange(y1, y2, self.sample_stride, dtype=np.int32)
        if xs.size == 0 or ys.size == 0:
            return None

        u, v = np.meshgrid(xs, ys)
        d_roi = depth[ys[:, None], xs[None, :]]

        if np.issubdtype(d_roi.dtype, np.integer):
            d_roi = d_roi.astype(np.float32) / self.depth_units_divisor
        else:
            d_roi = d_roi.astype(np.float32)

        if poly is not None:
            mask = self._poly_to_mask_roi(poly, x1, y1, xs.size, ys.size)
        else:
            mask = np.ones_like(d_roi, dtype=bool)

        valid = mask & np.isfinite(d_roi) & (d_roi > self.z_min) & (d_roi < self.z_max)
        if not np.any(valid):
            return None

        # Keep only the main mass
        if self.use_main_component:
            valid = self._keep_main_component(valid, d_roi)
            if not np.any(valid):
                return None

        z = d_roi[valid]
        uu = u[valid].astype(np.float32)
        vv = v[valid].astype(np.float32)

        x = (uu - self.cx) / self.fx * z
        y = (vv - self.cy) / self.fy * z

        # Camera optical frame convention:
        # x = right, y = down, z = forward
        pts = np.stack([x, y, z], axis=1)
        return pts

    def _poly_to_mask_roi(self, poly_xy, x1, y1, w_roi, h_roi):
        if len(poly_xy) < 3:
            return np.zeros((h_roi, w_roi), dtype=bool)

        pts = np.array(poly_xy, dtype=np.float32)
        pts[:, 0] = (pts[:, 0] - x1) / self.sample_stride
        pts[:, 1] = (pts[:, 1] - y1) / self.sample_stride
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        if self.mask_shrink_px > 0:
            k = max(1, self.mask_shrink_px)
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask.astype(bool)

    # --------------------------------------------------
    # Main-mass filtering
    # --------------------------------------------------
    def _largest_component(self, binary_mask):
        mask_u8 = binary_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        if num_labels <= 1:
            return binary_mask.copy()

        best_label = 0
        best_area = 0
        fallback_label = 0
        fallback_area = 0

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area > fallback_area:
                fallback_area = area
                fallback_label = label

            if area >= self.min_component_pixels and area > best_area:
                best_area = area
                best_label = label

        if best_label == 0:
            best_label = fallback_label

        if best_label == 0:
            return np.zeros_like(binary_mask, dtype=bool)

        return labels == best_label

    def _keep_main_component(self, valid_mask, d_roi):
        comp = self._largest_component(valid_mask)
        if not np.any(comp):
            return comp

        # Optional depth cleanup around the component's median depth
        if self.depth_band_m > 0.0:
            dvals = d_roi[comp]
            if dvals.size > 0:
                d_med = float(np.median(dvals))
                comp = comp & np.isfinite(d_roi) & (np.abs(d_roi - d_med) <= self.depth_band_m)
                comp = self._largest_component(comp)

        return comp

    # --------------------------------------------------
    # TF transform utilities
    # --------------------------------------------------
    def _transform_points(self, pts, from_frame, to_frame):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
        except TransformException:
            return None

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)

        pts_out = pts @ R.T + trans
        return pts_out

    def _quat_to_rot(self, x, y, z, w):
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)

    # --------------------------------------------------
    # PCA / OBB
    # --------------------------------------------------
    def _compute_pca_obb(self, pts: np.ndarray):
        if pts is None or pts.shape[0] < 10:
            return None

        centroid = np.mean(pts, axis=0)
        centered = pts - centroid

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)   # ascending
        order = np.argsort(eigvals)[::-1]        # descending
        eigvecs = eigvecs[:, order]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1.0

        local = centered @ eigvecs
        mins = np.min(local, axis=0)
        maxs = np.max(local, axis=0)
        extents = maxs - mins

        corners_local = np.array([
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
        ], dtype=np.float32)

        corners_world = corners_local @ eigvecs.T + centroid
        return centroid, eigvecs, extents, corners_world

    # --------------------------------------------------
    # Marker building
    # --------------------------------------------------
    def _build_markers(self, frame_id, stamp, det, centroid, eigvecs, extents, corners):
        markers = []

        axis_lengths = np.maximum(
            extents * 0.5,
            np.array([0.06, 0.05, 0.04], dtype=np.float32)
        )

        major_end = centroid + eigvecs[:, 0] * axis_lengths[0]
        middle_end = centroid + eigvecs[:, 1] * axis_lengths[1]
        minor_end = centroid + eigvecs[:, 2] * axis_lengths[2]

        markers.append(self._make_centroid_marker(frame_id, stamp, 0, centroid))
        markers.append(self._make_axis_marker(frame_id, stamp, 1, centroid, major_end, 1.0, 0.0, 0.0))
        markers.append(self._make_axis_marker(frame_id, stamp, 2, centroid, middle_end, 0.0, 1.0, 0.0))
        markers.append(self._make_axis_marker(frame_id, stamp, 3, centroid, minor_end, 0.0, 0.0, 1.0))
        markers.append(self._make_obb_marker(frame_id, stamp, 4, corners))
        markers.append(
            self._make_text_marker(
                frame_id,
                stamp,
                5,
                centroid,
                f'{det.class_name} {det.score:.2f}'
            )
        )

        return markers

    def _make_axis_marker(self, frame_id, stamp, marker_id, start, end, r, g, b):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'best_mask_pca'
        m.id = marker_id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        m.scale.y = 0.02
        m.scale.z = 0.03
        m.color = self._color(r, g, b, 1.0)
        m.points = [self._point(start), self._point(end)]
        return m

    def _make_centroid_marker(self, frame_id, stamp, marker_id, c):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'best_mask_pca'
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = self._point(c)
        m.pose.orientation.w = 1.0
        m.scale.x = 0.04
        m.scale.y = 0.04
        m.scale.z = 0.04
        m.color = self._color(1.0, 1.0, 0.0, 1.0)
        return m

    def _make_text_marker(self, frame_id, stamp, marker_id, pos, text):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'best_mask_pca'
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = self._point([pos[0], pos[1], pos[2] + 0.06])
        m.pose.orientation.w = 1.0
        m.scale.z = 0.05
        m.color = self._color(1.0, 1.0, 1.0, 1.0)
        m.text = text
        return m

    def _make_obb_marker(self, frame_id, stamp, marker_id, corners):
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = 'best_mask_pca'
        m.id = marker_id
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.006
        m.color = self._color(0.0, 1.0, 1.0, 1.0)

        for a, b in edges:
            m.points.append(self._point(corners[a]))
            m.points.append(self._point(corners[b]))

        return m

    def _point(self, xyz):
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2])
        return p

    def _color(self, r, g, b, a=1.0):
        c = ColorRGBA()
        c.r = float(r)
        c.g = float(g)
        c.b = float(b)
        c.a = float(a)
        return c


def main():
    rclpy.init()
    node = BestMaskMainMassPCA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()