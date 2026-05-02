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

# A much better starting pose than neutral
READY_POSE = {
    "waist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.55,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.15,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


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


def build_ready_q(model):
    q = pin.neutral(model)
    name_to_index = {name: i - 1 for i, name in enumerate(model.names) if i > 0}

    for name, val in READY_POSE.items():
        if name in name_to_index:
            q[name_to_index[name]] = val
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

        # World-aligned translational Jacobian
        J6 = pin.computeFrameJacobian(
            model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:3, :]  # position only

        # Damped pseudoinverse
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.solve(JJt + damping * np.eye(3), np.eye(3))

        dq_task = J_pinv @ err

        # Nullspace posture term: stay near ready pose
        N = np.eye(model.nv) - J_pinv @ J
        dq_posture = posture_gain * (q_nominal - q)

        dq = dq_task + N @ dq_posture

        # Limit update size
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq * (max_step / dq_norm)

        q = pin.integrate(model, q, dq)
        q = clamp_to_limits(model, q)

    return False, q, max_iters, err_norm


def print_joint_vector(model, q, title):
    print(f"\n{title}")
    for i, name in enumerate(model.names[1:], start=0):
        print(f"  {name:28s} {q[i]: .6f}")


def main():
    model = build_reduced_model()
    data = model.createData()

    print("Reduced joints:")
    for i, name in enumerate(model.names):
        print(f"  {i:2d}  {name}")

    q_ready = build_ready_q(model)

    current = fk(model, data, q_ready, FRAME_NAME)
    current_pos = current.translation.copy()

    print("\nCurrent pose at READY pose:")
    print("translation:", current_pos.T)
    print("rotation:\n", current.rotation)

    # Easy first target: small offset from current
    target_pos = current_pos + np.array([0.05, 0.00, 0.04])

    print("\nTarget position:")
    print(target_pos.T)

    ok, q_sol, iters, pos_err = solve_position_ik(
        model=model,
        data=data,
        frame_name=FRAME_NAME,
        q_init=q_ready,
        q_nominal=q_ready,
        target_pos=target_pos,
        max_iters=300,
        tol=2e-3,
        damping=1e-3,
        posture_gain=0.02,
        max_step=0.10,
    )

    final_pose = fk(model, data, q_sol, FRAME_NAME)

    print("\nIK success:", ok)
    print("iterations:", iters)
    print("final position error:", pos_err)

    print_joint_vector(model, q_ready, "READY joint values:")
    print_joint_vector(model, q_sol, "SOLVED joint values:")

    print("\nFinal EE position:")
    print(final_pose.translation.T)

    print("\nController-order positions:")
    print("[")
    for i, name in enumerate(model.names[1:], start=0):
        comma = "," if i < len(model.names[1:]) - 1 else ""
        print(f"  {q_sol[i]: .6f}{comma}  # {name}")
    print("]")


if __name__ == "__main__":
    main()