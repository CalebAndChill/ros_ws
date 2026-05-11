#!/usr/bin/env python3
"""
PCA + OBB -> pre-grasp point.

Listens to /best_mask_pca_markers (from best_mask_pca.py).
Computes a pre-grasp candidate as:
    pregrasp = centroid + sign * axis_unit * standoff
where:
    standoff   = min(half_extent_along_axis + clearance, max_standoff)
    half_extent = projection of OBB corners onto the chosen PCA axis

Frame-agnostic: outputs in whatever frame the input markers carry.
If best_mask_pca's output_frame is empty, that will be the camera optical
frame (camera_color_optical_frame).

Side selection (direction_sign):
    +1 -> always +axis_unit
    -1 -> always -axis_unit
     0 -> auto: pick the candidate with larger x (camera's right side).
          This is the right-hand-friendly default in camera optical frame.

Debug toggle:
    debug=True  -> show centroid, both candidates, chosen, arrow, label;
                   log every frame.
    debug=False -> show only centroid + chosen + arrow; log throttled.
"""
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray


class PcaObbToRightPregrasp(Node):
    def __init__(self):
        super().__init__("pca_obb_to_right_pregrasp")

        self.declare_parameter("input_topic", "/best_mask_pca_markers")
        self.declare_parameter("marker_topic", "/right_pregrasp_markers")
        self.declare_parameter("point_topic", "/right_pregrasp_point")

        # 1 = major (red), 2 = middle (green), 3 = minor (blue)
        self.declare_parameter("approach_axis_id", 2)

        # extra distance outside the object along chosen axis
        self.declare_parameter("clearance", 0.06)

        # cap stand-off so noisy OBB extents don't fling the target away
        self.declare_parameter("max_standoff", 0.20)

        # +1 -> +axis, -1 -> -axis, 0 -> auto (camera's right side, larger x)
        self.declare_parameter("direction_sign", 0)

        # verbose markers + per-frame logs
        self.declare_parameter("debug", True)
        self.declare_parameter("log_every_n", 30)  # used when debug=False

        self.input_topic = self.get_parameter("input_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.point_topic = self.get_parameter("point_topic").value
        self.approach_axis_id = int(self.get_parameter("approach_axis_id").value)
        self.clearance = float(self.get_parameter("clearance").value)
        self.max_standoff = float(self.get_parameter("max_standoff").value)
        self.direction_sign = int(self.get_parameter("direction_sign").value)
        self.debug = bool(self.get_parameter("debug").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        self.sub = self.create_subscription(
            MarkerArray, self.input_topic, self.cb_markers, 10
        )
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pub_point = self.create_publisher(PointStamped, self.point_topic, 10)

        self._cb_count = 0

        self.get_logger().info(f"Listening on {self.input_topic}")
        self.get_logger().info(f"Publishing markers on {self.marker_topic}")
        self.get_logger().info(f"Publishing chosen point on {self.point_topic}")
        self.get_logger().info(
            f"approach_axis_id={self.approach_axis_id} "
            f"(1=major/red, 2=middle/green, 3=minor/blue)"
        )
        self.get_logger().info(
            f"clearance={self.clearance:.3f} max_standoff={self.max_standoff:.3f} "
            f"direction_sign={self.direction_sign} debug={self.debug}"
        )

    def cb_markers(self, msg: MarkerArray):
        self._cb_count += 1

        centroid_marker = None
        axis_marker = None
        obb_marker = None

        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            if m.type == Marker.SPHERE and m.id == 0:
                centroid_marker = m
            elif m.type == Marker.ARROW and m.id == self.approach_axis_id:
                axis_marker = m
            elif m.type == Marker.LINE_LIST and m.id == 4:
                obb_marker = m

        if centroid_marker is None or axis_marker is None or obb_marker is None:
            return

        frame_id = centroid_marker.header.frame_id
        stamp = self.get_clock().now().to_msg()

        c = np.array([
            centroid_marker.pose.position.x,
            centroid_marker.pose.position.y,
            centroid_marker.pose.position.z,
        ], dtype=float)

        if len(axis_marker.points) < 2:
            return

        a0 = np.array(
            [axis_marker.points[0].x, axis_marker.points[0].y, axis_marker.points[0].z],
            dtype=float,
        )
        a1 = np.array(
            [axis_marker.points[1].x, axis_marker.points[1].y, axis_marker.points[1].z],
            dtype=float,
        )

        u = a1 - a0
        n = np.linalg.norm(u)
        if n < 1e-8:
            return
        u = u / n

        corners = self._extract_unique_corners(obb_marker.points)
        if corners.shape[0] < 4:
            return

        # extent of OBB along chosen axis
        proj = (corners - c) @ u
        extent = float(np.max(proj) - np.min(proj))
        half_extent = 0.5 * extent

        stand_off = min(half_extent + self.clearance, self.max_standoff)

        p_pos = c + u * stand_off   # +axis side
        p_neg = c - u * stand_off   # -axis side

        # side selection
        if self.direction_sign > 0:
            chosen = p_pos
            choice_str = "+axis (direction_sign>0)"
        elif self.direction_sign < 0:
            chosen = p_neg
            choice_str = "-axis (direction_sign<0)"
        else:
            # auto: in camera optical frame, larger x = camera's right side
            # which matches the robot's right arm approach side.
            if p_pos[0] >= p_neg[0]:
                chosen = p_pos
                choice_str = "+axis (auto: camera's right side)"
            else:
                chosen = p_neg
                choice_str = "-axis (auto: camera's right side)"

        should_log = self.debug or (self._cb_count % self.log_every_n == 0)
        if should_log:
            self.get_logger().info(
                f"frame={frame_id} centroid={c.round(4).tolist()} "
                f"u={u.round(4).tolist()} extent={extent:.4f} "
                f"standoff={stand_off:.4f}"
            )
            self.get_logger().info(
                f"p+={p_pos.round(4).tolist()} p-={p_neg.round(4).tolist()} "
                f"chosen={chosen.round(4).tolist()} [{choice_str}]"
            )

        self._publish_point(frame_id, stamp, chosen)
        self._publish_markers(frame_id, stamp, c, p_pos, p_neg, chosen)

    @staticmethod
    def _extract_unique_corners(points):
        uniq = {}
        for p in points:
            key = (round(p.x, 5), round(p.y, 5), round(p.z, 5))
            uniq[key] = [p.x, p.y, p.z]
        return np.array(list(uniq.values()), dtype=float)

    def _publish_point(self, frame_id, stamp, p):
        msg = PointStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.point.x = float(p[0])
        msg.point.y = float(p[1])
        msg.point.z = float(p[2])
        self.pub_point.publish(msg)

    def _publish_markers(self, frame_id, stamp, c, p_pos, p_neg, chosen):
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        # always: centroid (small yellow), chosen (small red), arrow centroid->chosen
        arr.markers.append(self._sphere(frame_id, stamp, 1, c,      1.0, 1.0, 0.0, 0.018))
        arr.markers.append(self._sphere(frame_id, stamp, 4, chosen, 1.0, 0.0, 0.0, 0.022))
        arr.markers.append(self._arrow(frame_id, stamp,  5, c, chosen, 0.0, 1.0, 1.0))

        # debug-only: both candidates and coord label
        if self.debug:
            arr.markers.append(self._sphere(frame_id, stamp, 2, p_pos, 0.6, 0.6, 0.6, 0.014))
            arr.markers.append(self._sphere(frame_id, stamp, 3, p_neg, 0.6, 0.6, 0.6, 0.014))

            label = f"[{chosen[0]:+.3f}, {chosen[1]:+.3f}, {chosen[2]:+.3f}]"
            arr.markers.append(self._text(frame_id, stamp, 6, chosen, label))

        self.pub_markers.publish(arr)

    def _sphere(self, frame_id, stamp, mid, xyz, r, g, b, s):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "right_pregrasp"
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.scale.x = s
        m.scale.y = s
        m.scale.z = s
        m.color.a = 1.0
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        return m

    def _arrow(self, frame_id, stamp, mid, p0, p1, r, g, b):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "right_pregrasp"
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.005   # shaft diameter
        m.scale.y = 0.012   # head diameter
        m.scale.z = 0.018   # head length
        m.color.a = 1.0
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)

        a = Point(); a.x, a.y, a.z = float(p0[0]), float(p0[1]), float(p0[2])
        b_pt = Point(); b_pt.x, b_pt.y, b_pt.z = float(p1[0]), float(p1[1]), float(p1[2])
        m.points = [a, b_pt]
        return m

    def _text(self, frame_id, stamp, mid, pos, text):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "right_pregrasp"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        # nudge label off the sphere; in camera optical frame y is down,
        # so -0.04 in y lifts the text visually upward.
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1]) - 0.04
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.z = 0.025
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.text = text
        return m


def main():
    rclpy.init()
    node = PcaObbToRightPregrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()