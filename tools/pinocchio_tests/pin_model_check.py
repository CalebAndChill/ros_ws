#!/usr/bin/env python3
import pinocchio as pin

URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    URDF, MESH_DIRS
)

print("model:", model.name)
print("nq:", model.nq, "nv:", model.nv)

print("\nJOINTS:")
for i, name in enumerate(model.names):
    print(f"{i:3d}  {name}")

print("\nLAST 80 FRAMES:")
for i, f in enumerate(model.frames[-80:]):
    print(f"{len(model.frames)-80+i:3d}  {f.name}")
