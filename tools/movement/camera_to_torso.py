#!/usr/bin/env python3
"""
camera_to_torso.py

Convert points from the camera optical frame to torso_link, using the same
G1 URDF the IK script loads. The chain torso_link -> head -> camera_link ->
camera optical frame is rigid on the 29-DOF G1, so this transform is a
constant - independent of pose.

Quick uses:
  # list candidate camera frames in the URDF
  python3 camera_to_torso.py --list

  # convert one point in camera frame to torso frame, print IK CLI args
  python3 camera_to_torso.py 0.1707 -0.1860 0.7027

  # explicit camera frame name (skips auto-detection)
  python3 camera_to_torso.py 0.17 -0.18 0.70 --camera-frame d435_optical_frame

Note: the RealSense ROS driver publishes points with header.frame_id =
'camera_color_optical_frame', but this URDF uses 'd435_optical_frame' for
the same physical optical pose. The script auto-resolves this by trying
common names and falling back to any *_optical_frame in the URDF.
"""

import argparse
import sys
import numpy as np
import pinocchio as pin


URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]

TORSO_FRAME = "torso_link"

# tried in order; first match wins
CAMERA_FRAME_CANDIDATES = (
    "camera_color_optical_frame",   # realsense ROS driver default
    "d435_color_optical_frame",
    "d435i_color_optical_frame",
    "d435_optical_frame",           # this URDF's name
    "d435i_optical_frame",
    "camera_optical_frame",
)

# IK script default workspace (override-able). Used only for a courtesy warning.
WS = dict(
    min_forward=0.05, max_forward=0.40,
    min_left=-0.40,   max_left=0.0,
    min_up=0.05,      max_up=0.60,
)


def load_model_with_fk():
    model, _, _ = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return model, data


def list_camera_frames(model):
    keywords = ("camera", "d435", "realsense", "color", "optical", "depth")
    return [f.name for f in model.frames
            if any(k in f.name.lower() for k in keywords)]


def resolve_camera_frame(model, requested):
    """Return (resolved_frame_name, was_explicit_match)."""
    if requested is not None:
        if model.existFrame(requested):
            return requested, True
        raise RuntimeError(
            f"Requested camera frame '{requested}' not in URDF.\n"
            f"Run with --list to see available camera frames."
        )

    # auto-detect: try the prioritised candidates first
    for name in CAMERA_FRAME_CANDIDATES:
        if model.existFrame(name):
            return name, False

    # last resort: any frame ending in _optical_frame
    for f in model.frames:
        if f.name.endswith("_optical_frame"):
            return f.name, False

    raise RuntimeError(
        "Could not auto-detect a camera optical frame in the URDF.\n"
        "Run with --list and then specify --camera-frame <name>."
    )


def static_transform(model, data, child_frame, parent_frame=TORSO_FRAME):
    """Return SE3 such that p_parent = T.act(p_child)."""
    if not model.existFrame(parent_frame):
        raise RuntimeError(f"Parent frame '{parent_frame}' not in URDF")
    if not model.existFrame(child_frame):
        raise RuntimeError(f"Child frame '{child_frame}' not in URDF")
    pid = model.getFrameId(parent_frame)
    cid = model.getFrameId(child_frame)
    return data.oMf[pid].actInv(data.oMf[cid])


def fmt_se3(T):
    R = T.rotation
    t = T.translation
    lines = []
    lines.append(f"  translation (origin of child frame, in parent frame, m):")
    lines.append(f"    [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]")
    lines.append(f"  rotation (p_parent = R @ p_child + t):")
    for i in range(3):
        lines.append(f"    [{R[i,0]:+.4f}, {R[i,1]:+.4f}, {R[i,2]:+.4f}]")
    return "\n".join(lines)


def workspace_warnings(fwd, left, up):
    msgs = []
    if fwd < WS["min_forward"] or fwd > WS["max_forward"]:
        msgs.append(f"forward {fwd:+.3f} outside [{WS['min_forward']}, {WS['max_forward']}]")
    if left < WS["min_left"] or left > WS["max_left"]:
        msgs.append(f"left {left:+.3f} outside [{WS['min_left']}, {WS['max_left']}] "
                    f"(positive y = robot's left; right-arm targets must be negative)")
    if up < WS["min_up"] or up > WS["max_up"]:
        msgs.append(f"up {up:+.3f} outside default [{WS['min_up']}, {WS['max_up']}] "
                    f"(IK script also accepts --ws-min-up 0.00 etc.)")
    return msgs


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("xyz", nargs="*", type=float,
                   help="Camera-frame point: x y z (meters)")
    p.add_argument("--camera-frame", default=None,
                   help="Override auto-detection (e.g. d435_optical_frame)")
    p.add_argument("--torso-frame", default=TORSO_FRAME)
    p.add_argument("--list", action="store_true",
                   help="List camera-related frames in the URDF and exit")
    args = p.parse_args()

    model, data = load_model_with_fk()

    if args.list:
        cams = list_camera_frames(model)
        print(f"Camera-related frames in {URDF}:")
        for c in cams:
            print(f"  {c}")
        try:
            resolved, _ = resolve_camera_frame(model, None)
            print(f"\nAuto-detected camera frame: {resolved}")
        except RuntimeError as exc:
            print(f"\n{exc}")
        return

    if len(args.xyz) != 3:
        print("Provide 3 floats: x y z (camera optical frame, meters).")
        print("Or use --list to inspect available camera frames.")
        sys.exit(1)

    cam_frame, explicit = resolve_camera_frame(model, args.camera_frame)
    if not explicit:
        print(f"Auto-detected camera frame: {cam_frame}")
        print(f"  (override with --camera-frame if this is wrong)\n")

    T = static_transform(model, data, cam_frame, args.torso_frame)

    print(f"Static transform: {cam_frame} -> {args.torso_frame}")
    print(fmt_se3(T))

    p_cam = np.array(args.xyz, dtype=float)
    p_torso = T.act(p_cam)
    fwd, left, up = float(p_torso[0]), float(p_torso[1]), float(p_torso[2])

    print(f"\nInput  (camera frame):  x={p_cam[0]:+.4f}, y={p_cam[1]:+.4f}, z={p_cam[2]:+.4f}")
    print(f"Output (torso frame):   forward={fwd:+.4f}, left={left:+.4f}, up={up:+.4f}")

    print(f"\nIK CLI args:")
    print(f"  --target-forward {fwd:.4f} --target-left {left:.4f} --target-up {up:.4f}")

    warns = workspace_warnings(fwd, left, up)
    if warns:
        print(f"\nNote on default IK workspace:")
        for w in warns:
            print(f"  - {w}")


if __name__ == "__main__":
    main()