#!/usr/bin/env python3
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

        # 2 = secondary axis (green), 3 = tertiary axis (blue)
        self.declare_parameter("approach_axis_id", 2)

        # extra distance outside the object along chosen axis
        self.declare_parameter("clearance", 0.06)

        # cap the total stand-off so bad OBB extents don't launch the target away
        self.declare_parameter("max_standoff", 0.20)

        # extra lift upward in torso_link to keep off the desk
        self.declare_parameter("lift_z", 0.03)

        # simple right-hand workspace clamp
        self.declare_parameter("x_min", 0.05)
        self.declare_parameter("x_max", 0.45)
        self.declare_parameter("y_min", -0.35)
        self.declare_parameter("y_max", -0.02)
        self.declare_parameter("z_min", -0.15)
        self.declare_parameter("z_max", 0.20)

        self.input_topic = self.get_parameter("input_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.point_topic = self.get_parameter("point_topic").value
        self.approach_axis_id = int(self.get_parameter("approach_axis_id").value)
        self.clearance = float(self.get_parameter("clearance").value)
        self.max_standoff = float(self.get_parameter("max_standoff").value)
        self.lift_z = float(self.get_parameter("lift_z").value)

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)

        self.sub = self.create_subscription(
            MarkerArray, self.input_topic, self.cb_markers, 10
        )
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pub_point = self.create_publisher(PointStamped, self.point_topic, 10)

        self.get_logger().info(f"Listening on {self.input_topic}")
        self.get_logger().info(f"Publishing markers on {self.marker_topic}")
        self.get_logger().info(f"Publishing chosen point on {self.point_topic}")
        self.get_logger().info(
            f"approach_axis_id={self.approach_axis_id}, "
            f"clearance={self.clearance:.3f}, "
            f"max_standoff={self.max_standoff:.3f}, "
            f"lift_z={self.lift_z:.3f}"
        )
        self.get_logger().info(
            f"workspace clamp: "
            f"x[{self.x_min:.2f}, {self.x_max:.2f}] "
            f"y[{self.y_min:.2f}, {self.y_max:.2f}] "
            f"z[{self.z_min:.2f}, {self.z_max:.2f}]"
        )

    def cb_markers(self, msg: MarkerArray):
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
            dtype=float
        )
        a1 = np.array(
            [axis_marker.points[1].x, axis_marker.points[1].y, axis_marker.points[1].z],
            dtype=float
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

        # cap stand-off so noisy extents cannot throw the target absurdly far away
        stand_off = min(half_extent + self.clearance, self.max_standoff)

        p1 = c + u * stand_off
        p2 = c - u * stand_off

        # small upward lift in torso frame
        p1[2] += self.lift_z
        p2[2] += self.lift_z

        # right-hand only: choose the candidate more on the robot's right
        # in torso_link, right side is more negative y
        chosen_raw = p1 if p1[1] < p2[1] else p2

        self.get_logger().info(f"centroid: {c}")
        self.get_logger().info(f"axis unit: {u}")
        self.get_logger().info(f"extent along axis: {extent:.6f}")
        self.get_logger().info(f"half_extent: {half_extent:.6f}")
        self.get_logger().info(f"stand_off: {stand_off:.6f}")
        self.get_logger().info(f"p1 raw: {p1}")
        self.get_logger().info(f"p2 raw: {p2}")
        self.get_logger().info(f"chosen raw: {chosen_raw}")

        chosen = chosen_raw.copy()

        # clamp into a sane right-hand workspace
        chosen[0] = np.clip(chosen[0], self.x_min, self.x_max)
        chosen[1] = np.clip(chosen[1], self.y_min, self.y_max)
        chosen[2] = np.clip(chosen[2], self.z_min, self.z_max)

        self.get_logger().info(f"chosen clamped: {chosen}")

        self._publish_point(frame_id, stamp, chosen)
        self._publish_markers(frame_id, stamp, c, p1, p2, chosen)

    def _extract_unique_corners(self, points):
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

    def _publish_markers(self, frame_id, stamp, c, p1, p2, chosen):
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        arr.markers.append(self._sphere(frame_id, stamp, 1, c, 1.0, 1.0, 0.0, 0.04))   # centroid yellow
        arr.markers.append(self._sphere(frame_id, stamp, 2, p1, 0.0, 0.0, 1.0, 0.05))  # candidate 1 blue
        arr.markers.append(self._sphere(frame_id, stamp, 3, p2, 0.0, 1.0, 0.0, 0.05))  # candidate 2 green
        arr.markers.append(self._sphere(frame_id, stamp, 4, chosen, 1.0, 0.0, 0.0, 0.06))  # chosen red

        arr.markers.append(self._line(frame_id, stamp, 5, c, p1, 0.0, 0.0, 1.0))
        arr.markers.append(self._line(frame_id, stamp, 6, c, p2, 0.0, 1.0, 0.0))
        arr.markers.append(self._line(frame_id, stamp, 7, c, chosen, 1.0, 0.0, 0.0))

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

    def _line(self, frame_id, stamp, mid, p0, p1, r, g, b):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = "right_pregrasp"
        m.id = mid
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.01
        m.color.a = 1.0
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)

        a = Point()
        a.x = float(p0[0])
        a.y = float(p0[1])
        a.z = float(p0[2])

        bpt = Point()
        bpt.x = float(p1[0])
        bpt.y = float(p1[1])
        bpt.z = float(p1[2])

        m.points = [a, bpt]
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