#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import cv2

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    from sensor_msgs import point_cloud2


class MaskToCloud(Node):
    def __init__(self):
        super().__init__('mask_to_cloud')

        # ---------- Parameters ----------
        self.declare_parameter('detections_topic', '/yolo/detections_3d')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('depth_units_divisor', 1000.0)     # 16UC1 mm -> meters
        self.declare_parameter('sample_stride', 4)                 # subsample pixels (↑ to reduce CPU)
        self.declare_parameter('mask_shrink_px', 0)                # erode mask N px (downsampled space)
        self.declare_parameter('publish_every_n', 1)               # publish 1 of every N detection msgs
        self.declare_parameter('min_score', 0.0)                   # drop low-score detections
        self.declare_parameter('allowed_classes', [])              # string[]; empty = all
        self.declare_parameter('max_points', 40000)                # cap points per message (0 = no cap)
        self.declare_parameter('z_min', 0.05)
        self.declare_parameter('z_max', 10.0)
        self.declare_parameter('use_roi', True)                    # compute only inside detection ROI
        self.declare_parameter('roi_expand_px', 0)                 # pad ROI in pixels (full-res space)

        det_topic = self.get_parameter('detections_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value

        self.div = float(self.get_parameter('depth_units_divisor').value)
        self.stride = int(self.get_parameter('sample_stride').value)
        self.mask_shrink = int(self.get_parameter('mask_shrink_px').value)
        self.pub_every = max(1, int(self.get_parameter('publish_every_n').value))
        self.min_score = float(self.get_parameter('min_score').value)
        self.allowed = set(self.get_parameter('allowed_classes').value or [])
        self.max_points = int(self.get_parameter('max_points').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.use_roi = bool(self.get_parameter('use_roi').value)
        self.roi_pad = int(self.get_parameter('roi_expand_px').value)

        q = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )

        from yolo_msgs.msg import DetectionArray  # runtime dep
        self.DetectionArray = DetectionArray

        self.bridge = CvBridge()
        self.depth = None
        self.depth_frame = 'camera_color_optical_frame'
        self.fx = self.fy = self.cx = self.cy = None
        self.msg_count = 0

        self.sub_info  = self.create_subscription(CameraInfo, info_topic, self.cb_info, 10)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.cb_depth, q)
        self.sub_det   = self.create_subscription(self.DetectionArray, det_topic, self.cb_det, 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/mask_cloud', 1)

        self.get_logger().info(
            f"MaskToCloud: det={det_topic} depth={depth_topic} info={info_topic} "
            f"(stride={self.stride}, shrink={self.mask_shrink}, pub_every={self.pub_every})"
        )

    # ---------- Callbacks ----------
    def cb_info(self, msg: CameraInfo):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def cb_depth(self, msg: Image):
        # track the frame so RViz doesn't need manual frame fiddling
        self.depth_frame = msg.header.frame_id or self.depth_frame
        self.depth = self.bridge.imgmsg_to_cv2(msg)  # expect 16UC1

    def _poly_to_mask_roi(self, poly_xy, x1, y1, w_roi, h_roi):
        """Rasterize polygon to a downsampled ROI mask (already divided by stride)."""
        if len(poly_xy) < 3:
            return np.zeros((h_roi, w_roi), dtype=bool)
        pts = np.array(poly_xy, dtype=np.float32)
        # shift to ROI and downsample
        pts[:, 0] = (pts[:, 0] - x1) / self.stride
        pts[:, 1] = (pts[:, 1] - y1) / self.stride
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if self.mask_shrink > 0:
            k = max(1, self.mask_shrink)
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        return mask.astype(bool)

    def cb_det(self, det_arr):
        if not isinstance(det_arr, self.DetectionArray):
            return
        if self.depth is None or self.fx is None:
            return

        self.msg_count += 1
        if (self.msg_count % self.pub_every) != 0:
            return

        depth = self.depth
        H, W = depth.shape[:2]
        all_pts = []

        for det in det_arr.detections:
            # filters to cut work early
            if det.score < self.min_score:
                continue
            if self.allowed and (det.class_name not in self.allowed):
                continue

            # ROI (full-res space)
            if det.mask and det.mask.data:
                poly = np.array([(p.x, p.y) for p in det.mask.data], dtype=np.float32)
                x1 = int(np.floor(np.min(poly[:, 0])))
                y1 = int(np.floor(np.min(poly[:, 1])))
                x2 = int(np.ceil (np.max(poly[:, 0])))
                y2 = int(np.ceil (np.max(poly[:, 1])))
            else:
                cx2d = int(det.bbox.center.position.x)
                cy2d = int(det.bbox.center.position.y)
                bw = int(det.bbox.size.x)
                bh = int(det.bbox.size.y)
                x1 = cx2d - bw // 2; x2 = cx2d + (bw - bw // 2)
                y1 = cy2d - bh // 2; y2 = cy2d + (bh - bh // 2)

            if self.use_roi:
                x1 = max(0, x1 - self.roi_pad); y1 = max(0, y1 - self.roi_pad)
                x2 = min(W, x2 + self.roi_pad); y2 = min(H, y2 + self.roi_pad)
            else:
                x1, y1, x2, y2 = 0, 0, W, H  # fallback to full frame

            if x2 <= x1 or y2 <= y1:
                continue

            # Downsampled ROI grid
            xs = np.arange(x1, x2, self.stride, dtype=np.int32)
            ys = np.arange(y1, y2, self.stride, dtype=np.int32)
            if xs.size == 0 or ys.size == 0:
                continue
            u, v = np.meshgrid(xs, ys)  # full-res coordinates at stride steps
            d_roi = depth[ys[:, None], xs[None, :]].astype(np.float32) / self.div  # meters

            # Mask in downsampled ROI
            if det.mask and det.mask.data:
                m = self._poly_to_mask_roi(poly, x1, y1, w_roi=xs.size, h_roi=ys.size)
            else:
                m = np.ones_like(d_roi, dtype=bool)

            valid = m & np.isfinite(d_roi) & (d_roi > self.z_min) & (d_roi < self.z_max)
            if not np.any(valid):
                continue

            z = d_roi[valid]
            uu = u[valid].astype(np.float32)
            vv = v[valid].astype(np.float32)
            x = (uu - self.cx) / self.fx * z
            y = (vv - self.cy) / self.fy * z
            pts = np.stack([x, y, z], axis=1)
            all_pts.append(pts)

        if not all_pts:
            return

        pts = np.concatenate(all_pts, axis=0)

        # Randomly cap points if needed (cheap)
        if self.max_points > 0 and pts.shape[0] > self.max_points:
            sel = np.random.choice(pts.shape[0], self.max_points, replace=False)
            pts = pts[sel]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.depth_frame
        cloud = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.pub_cloud.publish(cloud)


def main():
    rclpy.init()
    node = MaskToCloud()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
