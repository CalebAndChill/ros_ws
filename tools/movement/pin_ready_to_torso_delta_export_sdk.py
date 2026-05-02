#!/usr/bin/env python3
"""
Pinocchio IK exporter for real G1 SDK2 arm testing.

This script DOES NOT move the robot.

It:
1. Builds the same reduced right-arm-only Pinocchio model.
2. Solves position-only IK for right_wrist_yaw_link.
3. Prints the 7 right-arm joint targets.
4. Prints a copy-paste command for your real SDK2 movement script:
   ~/ros2_ws/tools/movement/g1_right_arm_preset.py

Why:
- Your Gazebo IK script publishes JointTrajectory to /whole_body_controller/joint_trajectory.
- That is correct for sim, but NOT what we want to directly run on the real robot.
- For the real robot, we first export the 7 right-arm values, then feed them into
  the safer SDK2 arm script with gradual-step, roll-first, per-joint KP, and logging.
"""

import argparse
import math
import os
from typing import Dict, List, Tuple

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

# Same nominal ready as your sim IK script.
READY_REDUCED = {
    "right_shoulder_pitch_joint": 0.10,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.20,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

SDK_ARG_NAMES = {
    "right_shoulder_pitch_joint": "right-shoulder-pitch",
    "right_shoulder_roll_joint": "right-shoulder-roll",
    "right_shoulder_yaw_joint": "right-shoulder-yaw",
    "right_elbow_joint": "right-elbow",
    "right_wrist_roll_joint": "right-wrist-roll",
    "right_wrist_pitch_joint": "right-wrist-pitch",
    "right_wrist_yaw_joint": "right-wrist-yaw",
}


def get_joint_q_index(model: pin.Model, joint_name: str) -> int:
    jid = model.getJointId(joint_name)
    if jid == 0 or jid >= len(model.joints):
        raise ValueError(f"Joint not found in model: {joint_name}")
    return model.joints[jid].idx_q


def set_joint_q(model: pin.Model, q: np.ndarray, joint_name: str, value: float) -> None:
    q_idx = get_joint_q_index(model, joint_name)
    q[q_idx] = float(value)


def get_joint_q(model: pin.Model, q: np.ndarray, joint_name: str) -> float:
    q_idx = get_joint_q_index(model, joint_name)
    return float(q[q_idx])


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
            raise RuntimeError(
                f"Reduced model is missing expected joint: {joint_name}"
            )

    return reduced_model


def build_ready_q_reduced(model: pin.Model) -> np.ndarray:
    q = pin.neutral(model)

    for joint_name, value in READY_REDUCED.items():
        set_joint_q(model, q, joint_name, value)

    return q


def fk(model: pin.Model, data: pin.Data, q: np.ndarray, frame_name: str) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    fid = model.getFrameId(frame_name)
    if fid >= len(model.frames):
        raise ValueError(f"Frame not found: {frame_name}")

    return data.oMf[fid]


def clamp_to_limits(model: pin.Model, q: np.ndarray) -> np.ndarray:
    q_clamped = q.copy()

    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()

    big = 1e10
    for i in range(len(q_clamped)):
        if lower[i] > -big and upper[i] < big:
            q_clamped[i] = np.clip(q_clamped[i], lower[i], upper[i])

    return q_clamped


def solve_position_ik(
    model: pin.Model,
    data: pin.Data,
    frame_name: str,
    q_init: np.ndarray,
    q_nominal: np.ndarray,
    target_pos_world: np.ndarray,
    max_iters: int = 300,
    tol: float = 2e-3,
    damping: float = 1e-3,
    posture_gain: float = 0.02,
    max_step: float = 0.10,
) -> Tuple[bool, np.ndarray, int, float]:
    fid = model.getFrameId(frame_name)
    q = q_init.copy()

    final_err_norm = float("inf")

    for i in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current = data.oMf[fid]
        pos = current.translation

        err = target_pos_world - pos
        err_norm = float(np.linalg.norm(err))
        final_err_norm = err_norm

        if err_norm < tol:
            return True, q, i, err_norm

        J6 = pin.computeFrameJacobian(
            model,
            data,
            q,
            fid,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J = J6[:3, :]

        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.solve(
            JJt + damping * np.eye(3),
            np.eye(3),
        )

        dq_task = J_pinv @ err

        # Keep solution near ready posture where possible.
        N = np.eye(model.nv) - J_pinv @ J
        dq_posture = posture_gain * (q_nominal - q)

        dq = dq_task + N @ dq_posture

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        q = pin.integrate(model, q, dq)
        q = clamp_to_limits(model, q)

    return False, q, max_iters, final_err_norm


def q_to_joint_dict(model: pin.Model, q: np.ndarray) -> Dict[str, float]:
    return {
        joint_name: get_joint_q(model, q, joint_name)
        for joint_name in CONTROL_JOINTS
    }


def print_joint_table(
    title: str,
    model: pin.Model,
    q_ready: np.ndarray,
    q_sol: np.ndarray,
) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)
    print(f"{'joint':<32} {'ready_q':>12} {'ik_q':>12} {'delta':>12}")
    print("-" * 84)

    max_delta = 0.0

    for joint_name in CONTROL_JOINTS:
        ready = get_joint_q(model, q_ready, joint_name)
        sol = get_joint_q(model, q_sol, joint_name)
        delta = sol - ready
        max_delta = max(max_delta, abs(delta))
        print(f"{joint_name:<32} {ready:+12.4f} {sol:+12.4f} {delta:+12.4f}")

    print("-" * 84)
    print(f"Largest IK-vs-ready joint delta: {max_delta:.4f} rad ({max_delta * 57.3:.1f} deg)")
    print("=" * 84)


def make_sdk_command(
    q_cmd: Dict[str, float],
    gradual_step: float,
    duration: float,
    hold: float,
    release: float,
    kp_test: float,
    shoulder_roll_kp: float,
    elbow_kp: float,
) -> str:
    lines: List[str] = []

    lines.append("python3 ~/ros2_ws/tools/movement/g1_right_arm_preset.py \\")
    lines.append("  --preset ready_small_out \\")
    lines.append("  --roll-first \\")
    lines.append(f"  --gradual-step {gradual_step:.2f} \\")

    for joint_name in CONTROL_JOINTS:
        sdk_arg = SDK_ARG_NAMES[joint_name]
        val = q_cmd[joint_name]
        lines.append(f"  --{sdk_arg} {val:+.5f} \\")

    lines.append(f"  --kp-test {kp_test:.1f} \\")
    lines.append(f"  --right-shoulder-roll-kp {shoulder_roll_kp:.1f} \\")
    lines.append(f"  --right-elbow-kp {elbow_kp:.1f} \\")
    lines.append(f"  --duration {duration:.1f} \\")
    lines.append(f"  --hold {hold:.1f} \\")
    lines.append(f"  --release {release:.1f} \\")
    lines.append("  --log-errors")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Export Pinocchio torso-frame right-arm IK as a safe SDK2 command."
    )

    parser.add_argument("--target-forward", type=float, default=0.12)
    parser.add_argument("--target-left", type=float, default=-0.18)
    parser.add_argument("--target-up", type=float, default=0.00)

    parser.add_argument("--gradual-step", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hold", type=float, default=8.0)
    parser.add_argument("--release", type=float, default=2.0)

    parser.add_argument("--kp-test", type=float, default=35.0)
    parser.add_argument("--right-shoulder-roll-kp", type=float, default=45.0)
    parser.add_argument("--right-elbow-kp", type=float, default=65.0)

    # For first real tests, we know right shoulder roll needs outward clearance.
    # More negative = more outward on your real robot tests.
    # If IK gives a shoulder roll less outward than this, we print a safer command too.
    parser.add_argument("--clearance-roll", type=float, default=-0.21)

    # Optional command compensation. Default 0.0 = exact IK command.
    # Later, if you prove actual elbow is always +0.05 high, you can try -0.05 carefully.
    parser.add_argument("--elbow-command-offset", type=float, default=0.0)

    args = parser.parse_args()

    print("=" * 84)
    print("PINOCCHIO RIGHT-ARM IK EXPORTER — NO ROBOT MOVEMENT")
    print("=" * 84)
    print(f"URDF: {URDF}")
    print(f"Torso frame: {TORSO_FRAME}")
    print(f"EE frame: {EE_FRAME}")
    print()
    print("Target is ABSOLUTE in torso frame:")
    print(f"  forward x = {args.target_forward:+.4f} m")
    print(f"  left    y = {args.target_left:+.4f} m")
    print(f"  up      z = {args.target_up:+.4f} m")
    print("=" * 84)

    model = build_reduced_model()
    data = model.createData()

    q_ready = build_ready_q_reduced(model)

    torso_pose = fk(model, data, q_ready, TORSO_FRAME)
    ee_ready_pose = fk(model, data, q_ready, EE_FRAME)
    ee_ready_in_torso = torso_pose.actInv(ee_ready_pose)

    target_torso = np.array(
        [
            args.target_forward,
            args.target_left,
            args.target_up,
        ],
        dtype=float,
    )

    target_world = torso_pose.rotation @ target_torso + torso_pose.translation

    print()
    print("READY wrist position in torso frame:")
    print(f"  x={ee_ready_in_torso.translation[0]:+.4f}, "
          f"y={ee_ready_in_torso.translation[1]:+.4f}, "
          f"z={ee_ready_in_torso.translation[2]:+.4f}")
    print()
    print("Requested target in torso frame:")
    print(f"  x={target_torso[0]:+.4f}, "
          f"y={target_torso[1]:+.4f}, "
          f"z={target_torso[2]:+.4f}")

    ok, q_sol, iters, pos_err = solve_position_ik(
        model=model,
        data=data,
        frame_name=EE_FRAME,
        q_init=q_ready,
        q_nominal=q_ready,
        target_pos_world=target_world,
        max_iters=300,
        tol=2e-3,
        damping=1e-3,
        posture_gain=0.02,
        max_step=0.10,
    )

    if not ok:
        print()
        print("IK FAILED")
        print(f"Iterations: {iters}")
        print(f"Final position error: {pos_err:.6f} m")
        print()
        print("Do not run anything on the real robot from this target.")
        raise SystemExit(1)

    final_pose = fk(model, data, q_sol, EE_FRAME)
    final_in_torso = torso_pose.actInv(final_pose)

    print()
    print("IK SUCCESS")
    print(f"Iterations: {iters}")
    print(f"Final solver error: {pos_err:.6f} m")
    print("Final wrist position in torso frame:")
    print(f"  x={final_in_torso.translation[0]:+.4f}, "
          f"y={final_in_torso.translation[1]:+.4f}, "
          f"z={final_in_torso.translation[2]:+.4f}")

    print_joint_table(
        title="RIGHT ARM IK RESULT",
        model=model,
        q_ready=q_ready,
        q_sol=q_sol,
    )

    exact_q = q_to_joint_dict(model, q_sol)

    # Optional elbow command compensation.
    # Default is 0, so this changes nothing unless you explicitly ask for it.
    if abs(args.elbow_command_offset) > 1e-9:
        exact_q["right_elbow_joint"] += args.elbow_command_offset

    print()
    print("=" * 84)
    print("EXACT IK SDK2 COMMAND")
    print("=" * 84)
    print(
        make_sdk_command(
            q_cmd=exact_q,
            gradual_step=args.gradual_step,
            duration=args.duration,
            hold=args.hold,
            release=args.release,
            kp_test=args.kp_test,
            shoulder_roll_kp=args.right_shoulder_roll_kp,
            elbow_kp=args.right_elbow_kp,
        )
    )

    # Safer first-real-test command:
    # if IK shoulder roll is not outward enough, bias it to known clearance roll.
    safe_q = dict(exact_q)
    ik_roll = safe_q["right_shoulder_roll_joint"]

    changed_for_clearance = False
    if ik_roll > args.clearance_roll:
        safe_q["right_shoulder_roll_joint"] = args.clearance_roll
        changed_for_clearance = True

    if changed_for_clearance:
        print()
        print("=" * 84)
        print("SAFER FIRST REAL TEST COMMAND — SHOULDER ROLL BIASED OUTWARD")
        print("=" * 84)
        print(
            f"IK shoulder_roll was {ik_roll:+.5f}, "
            f"but your real tests showed more outward clearance near "
            f"{args.clearance_roll:+.5f}."
        )
        print("This changes the exact IK pose slightly, but is safer for first physical testing.")
        print()
        print(
            make_sdk_command(
                q_cmd=safe_q,
                gradual_step=args.gradual_step,
                duration=args.duration,
                hold=args.hold,
                release=args.release,
                kp_test=args.kp_test,
                shoulder_roll_kp=args.right_shoulder_roll_kp,
                elbow_kp=args.right_elbow_kp,
            )
        )

    print()
    print("=" * 84)
    print("REAL-ROBOT TEST ORDER")
    print("=" * 84)
    print("1. First run the printed command at --gradual-step 0.25.")
    print("2. If clean, rerun with --gradual-step 0.50.")
    print("3. Then 0.75.")
    print("4. Only then 1.00.")
    print("5. Keep L2+B ready. Stop if fingers approach the leg/rack/wire.")
    print("=" * 84)


if __name__ == "__main__":
    main()