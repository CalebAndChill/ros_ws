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


class ReachBestRightHandIK(Node):
    """
    No-install approximate IK / reacher.

    Assumed detection convention from your setup:
      x = forward
      y = vertical
      z = left/right   (negative = robot's right side)

    This is NOT MoveIt/KDL. It is a lightweight geometric reacher:
      - pick highest-confidence 3D detection
      - build a pre-grasp target near that object
      - solve a simple 2-link arm geometry
      - publish a JointTrajectory to your existing whole_body_controller
    """

    def __init__(self):
        super().__init__("reach_best_right_hand_ik")

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

        # Same full-body joint order you already use
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

        # Ready pose from your existing working command
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

        # ------------------------------
        # Tunable geometry / gains
        # ------------------------------

        # Approx right shoulder position in torso_link frame
        self.SHOULDER_X = 0.02
        self.SHOULDER_Y = -0.18
        self.SHOULDER_Z = 0.22

        # Approx segment lengths
        self.UPPER_ARM = 0.23
        self.FOREARM = 0.24

        # Reach shaping
        self.PREGRASP_BACKOFF = 0.16
        self.TARGET_Z_OFFSET = 0.03
        self.MAX_REACH = 0.36
        self.MIN_REACH = 0.12

        # Keep torso fixed for now
        self.WAIST_YAW = 0.0
        self.WAIST_ROLL = 0.0
        self.WAIST_PITCH = 0.0

        # Arm mapping
        self.SHOULDER_PITCH_BIAS = 0.90
        self.SHOULDER_PITCH_GAIN = 0.70

        # Positive gain here means: object farther to robot's right (more negative z)
        # -> shoulder roll becomes more negative.
        self.SHOULDER_ROLL_BIAS = -0.20
        self.SHOULDER_ROLL_GAIN = 1.25

        self.SHOULDER_YAW_BIAS = 0.00
        self.SHOULDER_YAW_GAIN = 0.30

        self.WRIST_PITCH_BIAS = 0.00
        self.WRIST_PITCH_GAIN = -0.25

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

            # Ignore detections behind / basically inside torso
            if x <= 0.05:
                continue

            if score > best_score:
                best_score = score
                best = (x, y, z, score, getattr(det, "class_name", "unknown"))

        self.best_target = best

    def build_pose(self, obj_x: float, obj_y: float, obj_z: float):
        pose = dict(self.ready)

        # Detection convention:
        # x = forward, y = vertical, z = left/right
        tx = obj_x
        ty = obj_z
        tz = obj_y + self.TARGET_Z_OFFSET

        # Right-hand-only clamp: don't let the right hand chase targets across the body
        ty = min(ty, -0.02)

        # Don't let it target too close to chest
        tx = max(tx, 0.12)

        # Shoulder origin
        sx = self.SHOULDER_X
        sy = self.SHOULDER_Y
        sz = self.SHOULDER_Z

        # Vector shoulder -> target
        vx = tx - sx
        vy = ty - sy
        vz = tz - sz

        raw_dist = math.sqrt(vx * vx + vy * vy + vz * vz)
        if raw_dist < 1e-6:
            return pose

        # Pre-grasp backoff
        desired_dist = clamp(raw_dist - self.PREGRASP_BACKOFF, self.MIN_REACH, self.MAX_REACH)
        scale = desired_dist / raw_dist

        rx = vx * scale
        ry = vy * scale
        rz = vz * scale

        # Horizontal direction and elevation
        yaw = math.atan2(ry, rx)
        horiz = math.sqrt(rx * rx + ry * ry)
        elev = math.atan2(rz, horiz)

        # 2-link geometry in the reach plane
        d = math.sqrt(horiz * horiz + rz * rz)
        d = clamp(d, 0.05, self.UPPER_ARM + self.FOREARM - 1e-4)

        # Elbow flexion
        cos_internal = clamp(
            (self.UPPER_ARM * self.UPPER_ARM + self.FOREARM * self.FOREARM - d * d)
            / (2.0 * self.UPPER_ARM * self.FOREARM),
            -1.0,
            1.0,
        )
        internal_angle = math.acos(cos_internal)
        elbow_flex = math.pi - internal_angle

        # Shoulder lift angle
        cos_alpha = clamp(
            (self.UPPER_ARM * self.UPPER_ARM + d * d - self.FOREARM * self.FOREARM)
            / (2.0 * self.UPPER_ARM * d),
            -1.0,
            1.0,
        )
        alpha = math.acos(cos_alpha)
        shoulder_lift = elev + alpha

        # Keep torso fixed
        pose["waist_yaw_joint"] = self.WAIST_YAW
        pose["waist_roll_joint"] = self.WAIST_ROLL
        pose["waist_pitch_joint"] = self.WAIST_PITCH

        # Arm joints
        pose["right_shoulder_pitch_joint"] = clamp(
            self.SHOULDER_PITCH_BIAS - self.SHOULDER_PITCH_GAIN * shoulder_lift,
            -0.05, 1.10
        )

        pose["right_shoulder_roll_joint"] = clamp(
            self.SHOULDER_ROLL_BIAS + self.SHOULDER_ROLL_GAIN * yaw,
            -1.25, 0.40
        )

        pose["right_shoulder_yaw_joint"] = clamp(
            self.SHOULDER_YAW_BIAS + self.SHOULDER_YAW_GAIN * yaw,
            -0.60, 0.60
        )

        pose["right_elbow_joint"] = clamp(elbow_flex, 0.05, 1.35)

        pose["right_wrist_roll_joint"] = 0.0
        pose["right_wrist_pitch_joint"] = clamp(
            self.WRIST_PITCH_BIAS + self.WRIST_PITCH_GAIN * elev,
            -0.50, 0.50
        )
        pose["right_wrist_yaw_joint"] = 0.0

        self.get_logger().info(
            f"obj=({obj_x:.2f}, {obj_y:.2f}, {obj_z:.2f}) "
            f"target=({tx:.2f}, {ty:.2f}, {tz:.2f}) "
            f"reach=({rx:.2f}, {ry:.2f}, {rz:.2f}) "
            f"yaw={yaw:.2f} elev={elev:.2f} lift={shoulder_lift:.2f} elbow={elbow_flex:.2f}"
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
            f"Reaching toward best target: {name} | score={score:.2f} | "
            f"forward={x:.2f} vertical={y:.2f} lateral={z:.2f}"
        )


def main():
    rclpy.init()
    node = ReachBestRightHandIK()
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