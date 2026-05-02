#!/usr/bin/env python3
import math
import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

FRAME_NAME = "right_wrist_yaw_link"

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

READY_FULL = {
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

CONTROL_JOINTS = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

READY_REDUCED = {
    "waist_yaw_joint": 0.0,
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


def build_reduced_model():
    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
    q0 = pin.neutral(model)

    keep = set(CONTROL_JOINTS)
    lock_ids = []

    for j in model.names:
        if j == "universe":
            continue
        if j not in keep:
            jid = model.getJointId(j)
            if jid != 0:
                lock_ids.append(jid)

    reduced_model = pin.buildReducedModel(model, lock_ids, q0)
    return reduced_model


def build_ready_q_reduced(model):
    q = pin.neutral(model)
    for i, name in enumerate(model.names[1:], start=0):
        if name in READY_REDUCED:
            q[i] = READY_REDUCED[name]
    return q


def fk(model, data, q, frame_name):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(frame_name)
    return data.oMf[fid]


def clamp_to_limits(model, q):
    q_clamped = q.copy()
    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()

    big = 1e10
    for i in range(len(q_clamped)):
        if lower[i] > -big and upper[i] < big:
            q_clamped[i] = np.clip(q_clamped[i], lower[i], upper[i])

    return q_clamped


def solve_position_ik(
    model,
    data,
    frame_name,
    q_init,
    q_nominal,
    target_pos,
    max_iters=300,
    tol=2e-3,
    damping=1e-3,
    posture_gain=0.02,
    max_step=0.10,
):
    fid = model.getFrameId(frame_name)
    q = q_init.copy()

    for i in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current = data.oMf[fid]
        pos = current.translation
        err = target_pos - pos
        err_norm = np.linalg.norm(err)

        if err_norm < tol:
            return True, q, i, err_norm

        J6 = pin.computeFrameJacobian(
            model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:3, :]

        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.solve(JJt + damping * np.eye(3), np.eye(3))

        dq_task = J_pinv @ err
        N = np.eye(model.nv) - J_pinv @ J
        dq_posture = posture_gain * (q_nominal - q)

        dq = dq_task + N @ dq_posture

        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        q = pin.integrate(model, q, dq)
        q = clamp_to_limits(model, q)

    return False, q, max_iters, err_norm


def expand_reduced_to_full(q_reduced, reduced_model):
    full_positions = [READY_FULL[name] for name in FULL_JOINTS]

    reduced_map = {}
    for i, name in enumerate(reduced_model.names[1:], start=0):
        reduced_map[name] = q_reduced[i]

    for i, name in enumerate(FULL_JOINTS):
        if name in reduced_map:
            full_positions[i] = float(reduced_map[name])

    return full_positions


class PinReadyToFixedPregrasp(Node):
    def __init__(self):
        super().__init__("pin_ready_to_fixed_pregrasp")

        self.declare_parameter("controller_topic", "/whole_body_controller/joint_trajectory")
        self.declare_parameter("ready_duration", 3.0)
        self.declare_parameter("settle_time", 1.0)
        self.declare_parameter("move_duration", 3.0)

        # Fixed offset from wrist pose at READY.
        # This is in the Pinocchio model root frame, not torso_link yet.
        self.declare_parameter("target_dx", 0.05)
        self.declare_parameter("target_dy", 0.00)
        self.declare_parameter("target_dz", 0.04)

        self.controller_topic = self.get_parameter("controller_topic").value
        self.ready_duration = float(self.get_parameter("ready_duration").value)
        self.settle_time = float(self.get_parameter("settle_time").value)
        self.move_duration = float(self.get_parameter("move_duration").value)

        self.target_dx = float(self.get_parameter("target_dx").value)
        self.target_dy = float(self.get_parameter("target_dy").value)
        self.target_dz = float(self.get_parameter("target_dz").value)

        self.pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)

        self.model = build_reduced_model()
        self.data = self.model.createData()
        self.q_ready = build_ready_q_reduced(self.model)

        current = fk(self.model, self.data, self.q_ready, FRAME_NAME)
        self.current_pos = current.translation.copy()
        self.target_pos = self.current_pos + np.array(
            [self.target_dx, self.target_dy, self.target_dz], dtype=float
        )

        self.get_logger().info(f"Controller topic: {self.controller_topic}")
        self.get_logger().info(f"Using control frame: {FRAME_NAME}")
        self.get_logger().info(f"READY wrist position: {self.current_pos}")
        self.get_logger().info(f"Target pre-grasp position: {self.target_pos}")

        self.publish_ready()

        total_wait = self.ready_duration + self.settle_time
        self.timer = self.create_timer(total_wait, self.on_go_to_pregrasp)
        self.sent_pregrasp = False

    def publish_ready(self):
        traj = JointTrajectory()
        traj.joint_names = FULL_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = [READY_FULL[name] for name in FULL_JOINTS]
        pt.time_from_start = to_duration(self.ready_duration)

        traj.points = [pt]
        self.pub.publish(traj)
        self.get_logger().info("Published READY pose.")

    def on_go_to_pregrasp(self):
        if self.sent_pregrasp:
            return

        ok, q_sol, iters, pos_err = solve_position_ik(
            model=self.model,
            data=self.data,
            frame_name=FRAME_NAME,
            q_init=self.q_ready,
            q_nominal=self.q_ready,
            target_pos=self.target_pos,
            max_iters=300,
            tol=2e-3,
            damping=1e-3,
            posture_gain=0.02,
            max_step=0.10,
        )

        if not ok:
            self.get_logger().error(
                f"IK failed after {iters} iterations, final position error = {pos_err:.6f}"
            )
            self.timer.cancel()
            return

        final_pose = fk(self.model, self.data, q_sol, FRAME_NAME)
        self.get_logger().info(
            f"IK success in {iters} iterations, final error = {pos_err:.6f}"
        )
        self.get_logger().info(f"Final wrist position: {final_pose.translation}")

        full_positions = expand_reduced_to_full(q_sol, self.model)

        traj = JointTrajectory()
        traj.joint_names = FULL_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = full_positions
        pt.time_from_start = to_duration(self.move_duration)

        traj.points = [pt]
        self.pub.publish(traj)
        self.get_logger().info("Published PRE-GRASP trajectory.")

        self.sent_pregrasp = True
        self.timer.cancel()


def main():
    rclpy.init()
    node = PinReadyToFixedPregrasp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()