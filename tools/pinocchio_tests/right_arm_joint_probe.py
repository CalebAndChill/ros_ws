#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

FULL_JOINTS = [
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

# Known mild baseline that doesn't go crazy
BASELINE = {
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

def to_duration(seconds: float) -> Duration:
    sec = int(math.floor(seconds))
    nanosec = int((seconds - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)

class RightArmJointProbe(Node):
    def __init__(self):
        super().__init__("right_arm_joint_probe")

        self.declare_parameter("controller_topic", "/whole_body_controller/joint_trajectory")
        self.declare_parameter("joint_name", "right_shoulder_pitch_joint")
        self.declare_parameter("delta", 0.2)
        self.declare_parameter("move_duration", 3.0)

        self.controller_topic = self.get_parameter("controller_topic").value
        self.joint_name = self.get_parameter("joint_name").value
        self.delta = float(self.get_parameter("delta").value)
        self.move_duration = float(self.get_parameter("move_duration").value)

        if self.joint_name not in BASELINE:
            raise ValueError(f"Unknown joint_name: {self.joint_name}")

        self.pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)

        traj = JointTrajectory()
        traj.joint_names = FULL_JOINTS

        positions = [BASELINE[name] for name in FULL_JOINTS]
        idx = FULL_JOINTS.index(self.joint_name)
        positions[idx] += self.delta

        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = to_duration(self.move_duration)
        traj.points = [pt]

        self.get_logger().info(f"Publishing probe for {self.joint_name} with delta {self.delta:+.3f}")
        self.pub.publish(traj)

def main():
    rclpy.init()
    node = RightArmJointProbe()
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()