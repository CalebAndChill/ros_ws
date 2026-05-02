#!/usr/bin/env python3
import pinocchio as pin

URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]
FRAME_NAME = "right_wrist_yaw_link"

model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
data = model.createData()

q = pin.neutral(model)

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

fid = model.getFrameId(FRAME_NAME)
oMf = data.oMf[fid]

print("frame:", FRAME_NAME)
print("translation:", oMf.translation.T)
print("rotation:\n", oMf.rotation)
