#!/usr/bin/env python3
"""
FK checker for actual G1 right-arm joint values.

This script DOES NOT move the robot.

It takes actual right-arm q values from motor_state logs and computes where
right_wrist_yaw_link is in torso_link coordinates.

Use it after a real movement test to check:
  actual joint q -> actual wrist position in torso frame

This helps compare real reached pose against the IK target.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pinocchio as pin


URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

TORSO_FRAME = "torso_link"
EE_FRAME = "right_wrist_yaw_link"

CONTROL_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def get_joint_q_index(model: pin.Model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    if jid == 0 or jid >= len(model.joints):
        raise ValueError(f"Joint not found in model: {joint_name}")
    return model.joints[jid].idx_q


def set_joint_q(model: pin.Model, q: np.ndarray, joint_name: str, value: float) -> None:
    q_idx = get_joint_q_index(model, joint_name)
    q[q_idx] = float(value)


def build_reduced_model() -> pin.Model:
    if not os.path.exists(URDF):
        raise FileNotFoundError(f"URDF not found: {URDF}")

    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
    q0 = pin.neutral(model)

    keep = set(CONTROL_JOINTS)
    lock_ids = []

    for joint_name in model.names:
        if joint_name == "universe":
            continue
        if joint_name not in keep:
            jid = model.getJointId(joint_name)
            if jid != 0:
                lock_ids.append(jid)

    reduced_model = pin.buildReducedModel(model, lock_ids, q0)

    for joint_name in CONTROL_JOINTS:
        if not reduced_model.existJointName(joint_name):
            raise RuntimeError(f"Reduced model is missing joint: {joint_name}")

    return reduced_model


def fk(model: pin.Model, data: pin.Data, q: np.ndarray, frame_name: str) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    fid = model.getFrameId(frame_name)
    if fid >= len(model.frames):
        raise ValueError(f"Frame not found: {frame_name}")

    return data.oMf[fid]


def build_q_from_args(model: pin.Model, args: argparse.Namespace) -> np.ndarray:
    q = pin.neutral(model)

    values = {
        "right_shoulder_pitch_joint": args.right_shoulder_pitch,
        "right_shoulder_roll_joint": args.right_shoulder_roll,
        "right_shoulder_yaw_joint": args.right_shoulder_yaw,
        "right_elbow_joint": args.right_elbow,
        "right_wrist_roll_joint": args.right_wrist_roll,
        "right_wrist_pitch_joint": args.right_wrist_pitch,
        "right_wrist_yaw_joint": args.right_wrist_yaw,
    }

    for joint_name, value in values.items():
        set_joint_q(model, q, joint_name, value)

    return q


def main():
    parser = argparse.ArgumentParser(
        description="Compute FK wrist position from actual G1 right-arm q values."
    )

    # Defaults from your full real IK test hold.
    parser.add_argument("--right-shoulder-pitch", type=float, default=+0.3990)
    parser.add_argument("--right-shoulder-roll", type=float, default=-0.2187)
    parser.add_argument("--right-shoulder-yaw", type=float, default=+0.0810)
    parser.add_argument("--right-elbow", type=float, default=+0.0087)
    parser.add_argument("--right-wrist-roll", type=float, default=-0.0117)
    parser.add_argument("--right-wrist-pitch", type=float, default=-0.0358)
    parser.add_argument("--right-wrist-yaw", type=float, default=-0.0105)

    # Your IK target in torso frame.
    parser.add_argument("--target-forward", type=float, default=0.12)
    parser.add_argument("--target-left", type=float, default=-0.18)
    parser.add_argument("--target-up", type=float, default=0.00)

    args = parser.parse_args()

    model = build_reduced_model()
    data = model.createData()
    q_actual = build_q_from_args(model, args)

    torso_pose = fk(model, data, q_actual, TORSO_FRAME)
    wrist_pose = fk(model, data, q_actual, EE_FRAME)

    wrist_in_torso = torso_pose.actInv(wrist_pose).translation

    target = np.array(
        [
            args.target_forward,
            args.target_left,
            args.target_up,
        ],
        dtype=float,
    )

    error = wrist_in_torso - target
    error_norm = float(np.linalg.norm(error))

    print("=" * 84)
    print("FK ACTUAL RIGHT ARM — NO ROBOT MOVEMENT")
    print("=" * 84)
    print(f"URDF: {URDF}")
    print(f"Torso frame: {TORSO_FRAME}")
    print(f"EE frame: {EE_FRAME}")
    print()

    print("Input actual q values:")
    for joint_name, value in zip(
        CONTROL_JOINTS,
        [
            args.right_shoulder_pitch,
            args.right_shoulder_roll,
            args.right_shoulder_yaw,
            args.right_elbow,
            args.right_wrist_roll,
            args.right_wrist_pitch,
            args.right_wrist_yaw,
        ],
    ):
        print(f"  {joint_name:<32} {value:+.5f} rad")

    print()
    print("Actual wrist position in torso frame:")
    print(f"  forward x = {wrist_in_torso[0]:+.4f} m")
    print(f"  left    y = {wrist_in_torso[1]:+.4f} m")
    print(f"  up      z = {wrist_in_torso[2]:+.4f} m")

    print()
    print("Target wrist position in torso frame:")
    print(f"  forward x = {target[0]:+.4f} m")
    print(f"  left    y = {target[1]:+.4f} m")
    print(f"  up      z = {target[2]:+.4f} m")

    print()
    print("Position error actual - target:")
    print(f"  dx = {error[0]:+.4f} m  ({error[0] * 100:+.1f} cm)")
    print(f"  dy = {error[1]:+.4f} m  ({error[1] * 100:+.1f} cm)")
    print(f"  dz = {error[2]:+.4f} m  ({error[2] * 100:+.1f} cm)")
    print(f"  total error = {error_norm:.4f} m  ({error_norm * 100:.1f} cm)")

    print()
    if error_norm < 0.03:
        print("Result: pretty good for early real testing.")
    elif error_norm < 0.06:
        print("Result: usable for rough testing, but too loose for final grasping.")
    else:
        print("Result: large error. Do not trust this for object grasping yet.")

    print("=" * 84)


if __name__ == "__main__":
    main()