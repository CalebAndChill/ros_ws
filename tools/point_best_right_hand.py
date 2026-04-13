#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from yolo_msgs.msg import DetectionArray
except Exception as e:
    raise RuntimeError("This script expects yolo_msgs.msg.DetectionArray") from e


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PointBestRightHand(Node):
    def __init__(self):
        super().__init__("point_best_right_hand")

        self.declare_parameter("detections_topic", "/yolo/detections_3d")
        self.declare_parameter("cmd_topic", "/whole_body_controller/joint_trajectory")
        self.declare_parameter("min_score", 0.05)
        self.declare_parameter("publish_period", 0.4)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.min_score = float(self.get_parameter("min_score").value)
        self.publish_period = float(self.get_parameter("publish_period").value)

        self.pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)
        self.sub = self.create_subscription(
            DetectionArray, self.detections_topic, self.detections_cb, 10
        )
        self.timer = self.create_timer(self.publish_period, self.timer_cb)

        self.best_target: Optional[Tuple[float, float, float, float, str]] = None
        self.last_sent_key = None

        # ============================================================
        # TUNING MODIFIERS - EDIT THESE FIRST
        # ============================================================

        # Detection convention currently assumed:
        #   x = forward
        #   y = vertical
        #   z = left/right
        #
        # If the arm goes to the wrong side, first flip the sign of:
        #   R_SH_ROLL_GAIN
        #   R_SH_YAW_GAIN

        # Ignore detections with forward distance <= this
        self.MIN_FORWARD = 0.05

        # Distance -> extension mapping
        self.EXTEND_START_DIST = 0.35   # start straightening after this distance
        self.EXTEND_FULL_DIST = 1.10    # fully extended by this distance

        # Waist tuning
        self.WAIST_YAW_BIAS = 0.0
        self.WAIST_YAW_GAIN = 0.0       # set >0 if you want torso yaw again
        self.WAIST_YAW_MIN = -0.8
        self.WAIST_YAW_MAX = 0.8

        self.WAIST_PITCH_BIAS = 0.0
        self.WAIST_PITCH_GAIN = 0.0     # set >0 if you want torso pitch
        self.WAIST_PITCH_MIN = -0.20
        self.WAIST_PITCH_MAX = 0.20

        # Right shoulder pitch
        # Increase BIAS to lift arm overall
        # Increase GAIN to react more strongly to target height
        self.R_SH_PITCH_BIAS = 0.55
        self.R_SH_PITCH_PITCH_GAIN = 0.45
        self.R_SH_PITCH_EXTEND_GAIN = -0.20
        self.R_SH_PITCH_MIN = 0.05
        self.R_SH_PITCH_MAX = 1.00

        # Right shoulder roll
        # This is the main left/right alignment modifier
        self.R_SH_ROLL_BIAS = 1.0
        self.R_SH_ROLL_GAIN = -3.3
        self.R_SH_ROLL_MIN = -1.20
        self.R_SH_ROLL_MAX = 0.60

        # Right shoulder yaw
        self.R_SH_YAW_BIAS = 0.0
        self.R_SH_YAW_GAIN = 0.55
        self.R_SH_YAW_MIN = -0.60
        self.R_SH_YAW_MAX = 0.60

        # Right elbow
        # Lower values = straighter arm
        self.R_ELBOW_BIAS = 0.75
        self.R_ELBOW_EXTEND_GAIN = -0.65
        self.R_ELBOW_MIN = 0.08
        self.R_ELBOW_MAX = 0.80

        # Right wrist pitch
        self.R_WRIST_PITCH_BIAS = 0.0
        self.R_WRIST_PITCH_GAIN = -0.30
        self.R_WRIST_PITCH_MIN = -0.40
        self.R_WRIST_PITCH_MAX = 0.40

        # Optional small offsets if everything is consistently off
        self.YAW_OFFSET = 0.0
        self.PITCH_OFFSET = 0.0

        # ============================================================

        self.joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        self.ready = {
            "left_hip_pitch_joint": -0.15,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.35,
            "left_ankle_pitch_joint": -0.20,
            "left_ankle_roll_joint": 0.0,

            "right_hip_pitch_joint": -0.15,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.35,
            "right_ankle_pitch_joint": -0.20,
            "right_ankle_roll_joint": 0.0,

            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            "left_shoulder_pitch_joint": 0.55,
            "left_shoulder_roll_joint": 0.20,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.15,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            "right_shoulder_pitch_joint": 0.55,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.15,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

        self.get_logger().info(
            f"Listening on {self.detections_topic}, publishing to {self.cmd_topic}"
        )

    def detections_cb(self, msg: DetectionArray):
        best = None
        best_score = -1.0

        for det in msg.detections:
            score = float(getattr(det, "score", 0.0))
            if score < self.min_score:
                continue

            bbox3d = getattr(det, "bbox3d", None)
            if bbox3d is None or getattr(bbox3d, "center", None) is None:
                continue

            pos = bbox3d.center.position
            x = float(pos.x)
            y = float(pos.y)
            z = float(pos.z)

            if not all(math.isfinite(v) for v in [x, y, z]):
                continue

            if x <= self.MIN_FORWARD:
                continue

            if score > best_score:
                best_score = score
                best = (x, y, z, score, getattr(det, "class_name", "unknown"))

        self.best_target = best

    def build_pose(self, x: float, y: float, z: float):
        pose = dict(self.ready)

        # Detection convention for your setup
        forward = x
        vertical = y
        lateral = z

        yaw = math.atan2(lateral, forward) + self.YAW_OFFSET
        pitch = -math.atan2(vertical, math.sqrt(forward * forward + lateral * lateral)) + self.PITCH_OFFSET
        dist = math.sqrt(forward * forward + vertical * vertical + lateral * lateral)

        extend = (dist - self.EXTEND_START_DIST) / max(
            1e-6, (self.EXTEND_FULL_DIST - self.EXTEND_START_DIST)
        )
        extend = clamp(extend, 0.0, 1.0)

        # Waist
        pose["waist_yaw_joint"] = clamp(
            self.WAIST_YAW_BIAS + self.WAIST_YAW_GAIN * yaw,
            self.WAIST_YAW_MIN,
            self.WAIST_YAW_MAX,
        )

        pose["waist_pitch_joint"] = clamp(
            self.WAIST_PITCH_BIAS + self.WAIST_PITCH_GAIN * pitch,
            self.WAIST_PITCH_MIN,
            self.WAIST_PITCH_MAX,
        )

        pose["waist_roll_joint"] = 0.0

        # Right arm
        pose["right_shoulder_pitch_joint"] = clamp(
            self.R_SH_PITCH_BIAS
            + self.R_SH_PITCH_PITCH_GAIN * pitch
            + self.R_SH_PITCH_EXTEND_GAIN * extend,
            self.R_SH_PITCH_MIN,
            self.R_SH_PITCH_MAX,
        )

        pose["right_shoulder_roll_joint"] = clamp(
            self.R_SH_ROLL_BIAS + self.R_SH_ROLL_GAIN * yaw,
            self.R_SH_ROLL_MIN,
            self.R_SH_ROLL_MAX,
        )

        pose["right_shoulder_yaw_joint"] = clamp(
            self.R_SH_YAW_BIAS + self.R_SH_YAW_GAIN * yaw,
            self.R_SH_YAW_MIN,
            self.R_SH_YAW_MAX,
        )

        pose["right_elbow_joint"] = clamp(
            self.R_ELBOW_BIAS + self.R_ELBOW_EXTEND_GAIN * extend,
            self.R_ELBOW_MIN,
            self.R_ELBOW_MAX,
        )

        pose["right_wrist_roll_joint"] = 0.0
        pose["right_wrist_pitch_joint"] = clamp(
            self.R_WRIST_PITCH_BIAS + self.R_WRIST_PITCH_GAIN * pitch,
            self.R_WRIST_PITCH_MIN,
            self.R_WRIST_PITCH_MAX,
        )
        pose["right_wrist_yaw_joint"] = 0.0

        self.get_logger().info(
            f"target forward={forward:.2f} vertical={vertical:.2f} lateral={lateral:.2f} "
            f"dist={dist:.2f} extend={extend:.2f} "
            f"yaw={yaw:.2f} pitch={pitch:.2f} "
            f"waist_yaw={pose['waist_yaw_joint']:.2f} "
            f"r_sh_pitch={pose['right_shoulder_pitch_joint']:.2f} "
            f"r_sh_roll={pose['right_shoulder_roll_joint']:.2f} "
            f"r_sh_yaw={pose['right_shoulder_yaw_joint']:.2f} "
            f"r_elbow={pose['right_elbow_joint']:.2f}"
        )

        return pose

    def send_pose(self, pose_dict):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = [pose_dict[j] for j in self.joint_names]
        pt.time_from_start.sec = 1
        pt.time_from_start.nanosec = 0

        msg.points = [pt]
        self.pub.publish(msg)

    def timer_cb(self):
        if self.best_target is None:
            return

        x, y, z, score, name = self.best_target

        key = (round(x, 2), round(y, 2), round(z, 2), round(score, 2), name)
        if key == self.last_sent_key:
            return
        self.last_sent_key = key

        pose = self.build_pose(x, y, z)
        self.send_pose(pose)

        self.get_logger().info(
            f"Pointing at best target: {name} | score={score:.2f} | "
            f"forward={x:.2f} vertical={y:.2f} lateral={z:.2f}"
        )


def main():
    rclpy.init()
    node = PointBestRightHand()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()