#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray


class PcaCentroidToPregraspMarker(Node):
    def __init__(self):
        super().__init__("pca_centroid_to_pregrasp_marker")

        self.declare_parameter("input_topic", "/best_mask_pca_markers")
        self.declare_parameter("output_topic", "/pregrasp_target_markers")

        # Fixed simple right-hand pre-grasp offset in torso_link
        self.declare_parameter("offset_x", 0.00)
        self.declare_parameter("offset_y", -0.10)
        self.declare_parameter("offset_z", 0.02)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.offset_x = float(self.get_parameter("offset_x").value)
        self.offset_y = float(self.get_parameter("offset_y").value)
        self.offset_z = float(self.get_parameter("offset_z").value)

        self.sub = self.create_subscription(
            MarkerArray, self.input_topic, self.cb_markers, 10
        )
        self.pub = self.create_publisher(MarkerArray, self.output_topic, 10)

        self.get_logger().info(f"Listening on {self.input_topic}")
        self.get_logger().info(f"Publishing pre-grasp marker on {self.output_topic}")
        self.get_logger().info(
            f"Offset = [{self.offset_x:.3f}, {self.offset_y:.3f}, {self.offset_z:.3f}] in torso_link"
        )

    def cb_markers(self, msg: MarkerArray):
        centroid_marker = None

        for m in msg.markers:
            # In your PCA node, centroid is a SPHERE marker with id 0
            if m.type == Marker.SPHERE and m.id == 0:
                centroid_marker = m
                break

        if centroid_marker is None:
            return

        cx = centroid_marker.pose.position.x
        cy = centroid_marker.pose.position.y
        cz = centroid_marker.pose.position.z

        tx = cx + self.offset_x
        ty = cy + self.offset_y
        tz = cz + self.offset_z

        arr = MarkerArray()

        # clear old
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        # centroid marker (yellow)
        m1 = Marker()
        m1.header.frame_id = centroid_marker.header.frame_id
        m1.header.stamp = self.get_clock().now().to_msg()
        m1.ns = "pregrasp_debug"
        m1.id = 1
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        m1.pose.orientation.w = 1.0
        m1.pose.position.x = cx
        m1.pose.position.y = cy
        m1.pose.position.z = cz
        m1.scale.x = 0.04
        m1.scale.y = 0.04
        m1.scale.z = 0.04
        m1.color.a = 1.0
        m1.color.r = 1.0
        m1.color.g = 1.0
        m1.color.b = 0.0
        arr.markers.append(m1)

        # pre-grasp target marker (red)
        m2 = Marker()
        m2.header.frame_id = centroid_marker.header.frame_id
        m2.header.stamp = self.get_clock().now().to_msg()
        m2.ns = "pregrasp_debug"
        m2.id = 2
        m2.type = Marker.SPHERE
        m2.action = Marker.ADD
        m2.pose.orientation.w = 1.0
        m2.pose.position.x = tx
        m2.pose.position.y = ty
        m2.pose.position.z = tz
        m2.scale.x = 0.05
        m2.scale.y = 0.05
        m2.scale.z = 0.05
        m2.color.a = 1.0
        m2.color.r = 1.0
        m2.color.g = 0.0
        m2.color.b = 0.0
        arr.markers.append(m2)

        # line from centroid to pre-grasp
        m3 = Marker()
        m3.header.frame_id = centroid_marker.header.frame_id
        m3.header.stamp = self.get_clock().now().to_msg()
        m3.ns = "pregrasp_debug"
        m3.id = 3
        m3.type = Marker.LINE_LIST
        m3.action = Marker.ADD
        m3.pose.orientation.w = 1.0
        m3.scale.x = 0.01
        m3.color.a = 1.0
        m3.color.r = 0.0
        m3.color.g = 1.0
        m3.color.b = 1.0

        p1 = centroid_marker.pose.position
        p2 = Marker().pose.position
        p2.x = tx
        p2.y = ty
        p2.z = tz
        m3.points = [p1, p2]
        arr.markers.append(m3)

        self.pub.publish(arr)


def main():
    rclpy.init()
    node = PcaCentroidToPregraspMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
