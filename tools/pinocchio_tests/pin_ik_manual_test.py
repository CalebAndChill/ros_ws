#!/usr/bin/env python3
import numpy as np
import pinocchio as pin

URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

FRAME_NAME = "right_wrist_yaw_link"

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


def fk(model, data, q, frame_name):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId(frame_name)
    return data.oMf[fid]


def clamp_to_limits(model, q):
    q_clamped = q.copy()
    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()

    # Ignore absurd "infinite" limits
    big = 1e10
    for i in range(len(q_clamped)):
        if lower[i] > -big and upper[i] < big:
            q_clamped[i] = np.clip(q_clamped[i], lower[i], upper[i])

    return q_clamped


def solve_ik(model, data, frame_name, q_init, target_se3,
             max_iters=400,
             dt=0.2,
             damping=1e-4,
             pos_tol=1e-3,
             rot_tol=2e-2):
    fid = model.getFrameId(frame_name)
    q = q_init.copy()

    for i in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current = data.oMf[fid]

        # Error in LOCAL frame of current EE pose
        err_se3 = current.actInv(target_se3)
        err = pin.log6(err_se3).vector  # [vx, vy, vz, wx, wy, wz]

        pos_err = np.linalg.norm(err[:3])
        rot_err = np.linalg.norm(err[3:])

        if pos_err < pos_tol and rot_err < rot_tol:
            return True, q, i, pos_err, rot_err

        J = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL)

        # Damped least squares
        JJt = J @ J.T
        v = -J.T @ np.linalg.solve(JJt + damping * np.eye(6), err)

        q = pin.integrate(model, q, v * dt)
        q = clamp_to_limits(model, q)

    return False, q, max_iters, pos_err, rot_err


def main():
    model = build_reduced_model()
    data = model.createData()

    print("Reduced joints:")
    for i, name in enumerate(model.names):
        print(f"  {i:2d}  {name}")

    q0 = pin.neutral(model)

    current = fk(model, data, q0, FRAME_NAME)
    print("\nCurrent pose at neutral:")
    print("translation:", current.translation.T)
    print("rotation:\n", current.rotation)

    # First easy reachable target:
    # keep same orientation, move slightly forward and up
    target_translation = current.translation + np.array([0.06, 0.00, 0.05])

    # Keep same orientation for the first test
    target_rotation = current.rotation.copy()

    target = pin.SE3(target_rotation, target_translation)

    print("\nTarget pose:")
    print("translation:", target.translation.T)
    print("rotation:\n", target.rotation)

    ok, q_sol, iters, pos_err, rot_err = solve_ik(
        model=model,
        data=data,
        frame_name=FRAME_NAME,
        q_init=q0,
        target_se3=target,
        max_iters=500,
        dt=0.2,
        damping=1e-4,
        pos_tol=1e-3,
        rot_tol=2e-2,
    )

    final_pose = fk(model, data, q_sol, FRAME_NAME)

    print("\nIK success:", ok)
    print("iterations:", iters)
    print("final position error:", pos_err)
    print("final rotation error:", rot_err)

    print("\nSolved joint values:")
    # model.names = ["universe", joint1, joint2, ...]
    for i, name in enumerate(model.names[1:], start=0):
        print(f"  {name:28s} {q_sol[i]: .6f}")

    print("\nFinal EE pose:")
    print("translation:", final_pose.translation.T)
    print("rotation:\n", final_pose.rotation)

    print("\nController-order positions:")
    print("[")
    for i, name in enumerate(model.names[1:], start=0):
        comma = "," if i < len(model.names[1:]) - 1 else ""
        print(f"  {q_sol[i]: .6f}{comma}  # {name}")
    print("]")


if __name__ == "__main__":
    main()