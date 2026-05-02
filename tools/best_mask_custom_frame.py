#!/usr/bin/env python3
import copy
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    from sensor_msgs import point_cloud2


class BestMaskCustomFrame(Node):
    """
    Converts raw PCA/cloud output from camera optical frame into a simple
    IK-friendly custom frame.

    Mapping used:
        x_out = z_in + forward_offset
        y_out = -x_in + left_offset
        z_out = -y_in + up_offset

    This is NOT a true URDF torso_link transform.
    It is a calibrated camera->IK frame mapping.
    """

    def __init__(self):
        super().__init__("best_mask_custom_frame")

        self.declare_parameter("input_cloud_topic", "/best_mask_cloud")
        self.declare_parameter("input_markers_topic", "/best_mask_pca_markers")
        self.declare_parameter("output_cloud_topic", "/best_mask_cloud_custom")
        self.declare_parameter("output_markers_topic", "/best_mask_pca_markers_custom")
        self.declare_parameter("output_centroid_topic", "/best_mask_centroid_custom")

        self.declare_parameter("source_frame_expected", "d435_optical_frame")
        self.declare_parameter("output_frame", "ik_calibrated")

        # Calibrated from your example:
        # raw optical point:
        #   x=1.2803955078125
        #   y=0.6037914156913757
        #   z=0.5709306573867798
        #
        # desired IK target:
        #   forward=0.29
        #   left=-0.15
        #   up=0.05
        #
        # giving:
        #   x_out = z_in - 0.2809306574
        #   y_out = -x_in + 1.1303955078
        #   z_out = -y_in + 0.6537914157
        self.declare_parameter("forward_offset", -0.2809306574)
        self.declare_parameter("left_offset", 1.1303955078)
        self.declare_parameter("up_offset", 0.6537914157)

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.input_markers_topic = self.get_parameter("input_markers_topic").value
        self.output_cloud_topic = self.get_parameter("output_cloud_topic").value
        self.output_markers_topic = self.get_parameter("output_markers_topic").value
        self.output_centroid_topic = self.get_parameter("output_centroid_topic").value

        self.source_frame_expected = str(self.get_parameter("source_frame_expected").value)
        self.output_frame = str(self.get_parameter("output_frame").value)

        self.forward_offset = float(self.get_parameter("forward_offset").value)
        self.left_offset = float(self.get_parameter("left_offset").value)
        self.up_offset = float(self.get_parameter("up_offset").value)

        self.pub_cloud = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.output_markers_topic, 10)
        self.pub_centroid = self.create_publisher(PointStamped, self.output_centroid_topic, 10)

        self.sub_cloud = self.create_subscription(
            PointCloud2, self.input_cloud_topic, self.cb_cloud, 10
        )
        self.sub_markers = self.create_subscription(
            MarkerArray, self.input_markers_topic, self.cb_markers, 10
        )

        self.get_logger().info(
            "BestMaskCustomFrame running.\n"
            f"  input_cloud_topic:    {self.input_cloud_topic}\n"
            f"  input_markers_topic:  {self.input_markers_topic}\n"
            f"  output_cloud_topic:   {self.output_cloud_topic}\n"
            f"  output_markers_topic: {self.output_markers_topic}\n"
            f"  output_centroid_topic:{self.output_centroid_topic}\n"
            f"  source_frame_expected:{self.source_frame_expected}\n"
            f"  output_frame:         {self.output_frame}\n"
            f"  mapping: x=z+({self.forward_offset:.6f}), "
            f"y=-x+({self.left_offset:.6f}), "
            f"z=-y+({self.up_offset:.6f})"
        )

    def make_point(self, xyz):
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2])
        return p

    def transform_xyz(self, xyz):
        x_in = float(xyz[0])
        y_in = float(xyz[1])
        z_in = float(xyz[2])

        x_out = z_in + self.forward_offset
        y_out = -x_in + self.left_offset
        z_out = -y_in + self.up_offset

        return np.array([x_out, y_out, z_out], dtype=np.float32)

    def transform_points_array(self, pts):
        pts = np.asarray(pts, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        x_in = pts[:, 0]
        y_in = pts[:, 1]
        z_in = pts[:, 2]

        x_out = z_in + self.forward_offset
        y_out = -x_in + self.left_offset
        z_out = -y_in + self.up_offset

        return np.stack([x_out, y_out, z_out], axis=1).astype(np.float32)

    def extract_xyz_array(self, msg: PointCloud2):
        pts = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )

        arr = np.asarray(list(pts))
        if arr.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # structured dtype case
        if arr.dtype.fields is not None:
            arr = np.stack(
                [
                    arr["x"].astype(np.float32),
                    arr["y"].astype(np.float32),
                    arr["z"].astype(np.float32),
                ],
                axis=1,
            )
            return arr

        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        return arr

    def cb_cloud(self, msg: PointCloud2):
        in_frame = msg.header.frame_id

        if self.source_frame_expected and in_frame != self.source_frame_expected:
            self.get_logger().warn(
                f"Cloud frame mismatch: expected {self.source_frame_expected}, got {in_frame}"
            )

        xyz = self.extract_xyz_array(msg)
        if xyz.shape[0] == 0:
            return

        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        if xyz.shape[0] == 0:
            return

        xyz_out = self.transform_points_array(xyz)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.output_frame

        out_msg = point_cloud2.create_cloud_xyz32(header, xyz_out.tolist())
        self.pub_cloud.publish(out_msg)

    def cb_markers(self, msg: MarkerArray):
        out = MarkerArray()
        centroid_done = False

        for m in msg.markers:
            nm = copy.deepcopy(m)
            nm.header.frame_id = self.output_frame

            if m.action == Marker.DELETEALL:
                out.markers.append(nm)
                continue

            # Arrow / line list style markers
            if len(m.points) > 0:
                nm.points = []
                for p in m.points:
                    xyz = self.transform_xyz([p.x, p.y, p.z])
                    nm.points.append(self.make_point(xyz))

            # Sphere / text / cube style markers
            else:
                xyz = self.transform_xyz(
                    [m.pose.position.x, m.pose.position.y, m.pose.position.z]
                )
                nm.pose.position = self.make_point(xyz)

                # publish centroid from first sphere marker
                if not centroid_done and m.type == Marker.SPHERE and m.id == 0:
                    pt = PointStamped()
                    pt.header.frame_id = self.output_frame
                    pt.header.stamp = m.header.stamp
                    pt.point = self.make_point(xyz)
                    self.pub_centroid.publish(pt)
                    centroid_done = True

            out.markers.append(nm)

        self.pub_markers.publish(out)


def main():
    rclpy.init()
    node = BestMaskCustomFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()