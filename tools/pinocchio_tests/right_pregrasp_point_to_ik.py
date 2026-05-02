#!/usr/bin/env python3
import math
import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

TORSO_FRAME = "torso_link"
EE_FRAME = "right_wrist_yaw_link"

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

    "right_shoulder_pitch_joint": 0.1,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.2,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

CONTROL_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

READY_REDUCED = {
    "right_shoulder_pitch_joint": 0.1,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.2,
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
    target_pos_world,
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
        err = target_pos_world - pos
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

    # keep torso fixed
    full_positions[FULL_JOINTS.index("waist_yaw_joint")] = 0.0
    full_positions[FULL_JOINTS.index("waist_roll_joint")] = 0.0
    full_positions[FULL_JOINTS.index("waist_pitch_joint")] = 0.0

    return full_positions


class RightPregraspPointToIK(Node):
    def __init__(self):
        super().__init__("right_pregrasp_point_to_ik")

        self.declare_parameter("input_topic", "/right_pregrasp_point")
        self.declare_parameter("controller_topic", "/whole_body_controller/joint_trajectory")
        self.declare_parameter("ready_duration", 3.0)
        self.declare_parameter("settle_time", 1.0)
        self.declare_parameter("move_duration", 3.0)
        self.declare_parameter("solve_once", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.controller_topic = self.get_parameter("controller_topic").value
        self.ready_duration = float(self.get_parameter("ready_duration").value)
        self.settle_time = float(self.get_parameter("settle_time").value)
        self.move_duration = float(self.get_parameter("move_duration").value)
        self.solve_once = bool(self.get_parameter("solve_once").value)

        self.pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)
        self.sub = self.create_subscription(PointStamped, self.input_topic, self.cb_point, 10)

        self.model = build_reduced_model()
        self.data = self.model.createData()
        self.q_ready = build_ready_q_reduced(self.model)

        self.torso_pose = fk(self.model, self.data, self.q_ready, TORSO_FRAME)
        self.ee_pose = fk(self.model, self.data, self.q_ready, EE_FRAME)

        self.target_torso = None
        self.sent_ik = False
        self.ready_sent = False

        self.get_logger().info(f"Listening on {self.input_topic}")
        self.get_logger().info(f"Publishing trajectories to {self.controller_topic}")
        self.get_logger().info(f"Using torso frame {TORSO_FRAME} and EE frame {EE_FRAME}")

        self.publish_ready()

        total_wait = self.ready_duration + self.settle_time
        self.timer = self.create_timer(total_wait, self.try_solve)

    def publish_ready(self):
        traj = JointTrajectory()
        traj.joint_names = FULL_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = [READY_FULL[name] for name in FULL_JOINTS]
        pt.time_from_start = to_duration(self.ready_duration)

        traj.points = [pt]
        self.pub.publish(traj)
        self.ready_sent = True
        self.get_logger().info("Published READY pose.")

    def cb_point(self, msg: PointStamped):
        if msg.header.frame_id != TORSO_FRAME:
            self.get_logger().warn(
                f"Expected point in {TORSO_FRAME}, got {msg.header.frame_id}"
            )
            return

        self.target_torso = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        self.get_logger().info(f"Received pre-grasp point in torso frame: {self.target_torso}")

    def try_solve(self):
        if self.solve_once and self.sent_ik:
            return

        if self.target_torso is None:
            self.get_logger().info("Waiting for /right_pregrasp_point ...")
            return

        target_world = self.torso_pose.rotation @ self.target_torso + self.torso_pose.translation

        self.get_logger().info(f"Target in world/model frame: {target_world}")

        ok, q_sol, iters, pos_err = solve_position_ik(
            model=self.model,
            data=self.data,
            frame_name=EE_FRAME,
            q_init=self.q_ready,
            q_nominal=self.q_ready,
            target_pos_world=target_world,
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
            return

        final_pose = fk(self.model, self.data, q_sol, EE_FRAME)
        final_torso = self.torso_pose.actInv(final_pose)

        self.get_logger().info(f"IK success in {iters} iterations, final error = {pos_err:.6f}")
        self.get_logger().info(f"Final wrist position in torso frame: {final_torso.translation}")

        full_positions = expand_reduced_to_full(q_sol, self.model)

        traj = JointTrajectory()
        traj.joint_names = FULL_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = full_positions
        pt.time_from_start = to_duration(self.move_duration)

        traj.points = [pt]
        self.pub.publish(traj)
        self.get_logger().info("Published trajectory to pre-grasp point.")

        self.sent_ik = True
        if self.solve_once:
            self.timer.cancel()


def main():
    rclpy.init()
    node = RightPregraspPointToIK()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
