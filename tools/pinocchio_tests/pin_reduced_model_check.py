#!/usr/bin/env python3
import pinocchio as pin

URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
q0 = pin.neutral(model)

keep = {
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}

lock_ids = []
for j in model.names:
    if j == "universe":
        continue
    if j not in keep:
        jid = model.getJointId(j)
        if jid != 0:
            lock_ids.append(jid)

reduced_model = pin.buildReducedModel(model, lock_ids, q0)

print("reduced model nq:", reduced_model.nq, "nv:", reduced_model.nv)
print("\nREDUCED JOINTS:")
for i, name in enumerate(reduced_model.names):
    print(f"{i:3d}  {name}")

