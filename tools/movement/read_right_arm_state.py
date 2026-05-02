#!/usr/bin/env python3
"""
G1 read-only right-side state monitor: arm + hand in one script.

Subscribes to:
    rt/lowstate              -> 7 right-arm joints (LowState_)
    rt/dex3/right/state      -> 7 Dex3 hand joints + 9 pressure modules
    rt/dex3/left/state       -> (optional, with --side both)

Does NOT publish any commands. Safe to run on the real robot before
attempting any motion.

Intended location:
    ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py

Usage:
    python3 ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py
    python3 ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py --rate 5
    python3 ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py --side both
    python3 ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py --no-hand
    python3 ~/ros2_ws/tools/movement/read_right_arm_and_hand_state.py --no-pressure

Exit with Ctrl-C. Nothing on the robot moves either way.
"""

import os
import sys
import site

# -------------------------------------------------------------------
# Let plain Python see your Unitree SDK repo + venv packages
# (matches the bootstrap used in your existing SDK2 scripts)
# -------------------------------------------------------------------
HOME = os.path.expanduser("~")
UNITREE_REPO = os.path.join(HOME, "unitree_sdk2_python")
VENV_SITE = os.path.join(
    HOME,
    "unitree_sdk2_venv",
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)

if os.path.isdir(UNITREE_REPO) and UNITREE_REPO not in sys.path:
    sys.path.insert(0, UNITREE_REPO)

if os.path.isdir(VENV_SITE):
    site.addsitedir(VENV_SITE)
# -------------------------------------------------------------------

import argparse
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_


# ---- Right-arm joint indices in the LowState motor_state array ----
class G1JointIndex(IntEnum):
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


RIGHT_ARM = [
    ("right_shoulder_pitch", G1JointIndex.RightShoulderPitch),
    ("right_shoulder_roll",  G1JointIndex.RightShoulderRoll),
    ("right_shoulder_yaw",   G1JointIndex.RightShoulderYaw),
    ("right_elbow",          G1JointIndex.RightElbow),
    ("right_wrist_roll",     G1JointIndex.RightWristRoll),
    ("right_wrist_pitch",    G1JointIndex.RightWristPitch),
    ("right_wrist_yaw",      G1JointIndex.RightWristYaw),
]


# ---- Dex3 hand joint labels (confirmed by physical wiggle test) ----
# joint 0     : thumb abduction (rotates thumb across palm; barely moves
#               in resting pose — that's how it was identified)
# joints 1,2  : thumb flexion (the two joints that curl the thumb)
# joints 3,4  : left-side finger (Caleb's perspective)
# joints 5,6  : right-side finger (Caleb's perspective)
# Rename to index/middle once viewing convention is confirmed.
HAND_JOINT_LABELS = [
    "thumb_abduction",
    "thumb_flex_0",
    "thumb_flex_1",
    "finger_left_0",
    "finger_left_1",
    "finger_right_0",
    "finger_right_1",
]

NUM_HAND_JOINTS = 7
NUM_PRESS_MODULES = 9


# -------------------------------------------------------------------
# Monitors
# -------------------------------------------------------------------
class LowStateMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.first_received = False
        self.msg_count = 0

    def handler(self, msg: LowState_):
        with self.lock:
            self.latest = msg
            self.msg_count += 1
            if not self.first_received:
                self.first_received = True


class HandStateMonitor:
    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.latest = None
        self.first_received = False
        self.msg_count = 0

    def handler(self, msg: HandState_):
        with self.lock:
            self.latest = msg
            self.msg_count += 1
            if not self.first_received:
                self.first_received = True


# -------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------
def fmt_joint(label: str, q: float, dq: float) -> str:
    return f"  {label:24s}  q = {q:+8.4f} rad   dq = {dq:+8.4f} rad/s"


def print_arm_snapshot(msg: LowState_) -> None:
    print("  ARM (right):")
    for label, idx in RIGHT_ARM:
        ms = msg.motor_state[idx]
        print(fmt_joint(label, ms.q, ms.dq))


def print_hand_snapshot(name: str, msg: HandState_, show_pressure: bool) -> None:
    print(f"  HAND ({name}):")
    try:
        for i in range(NUM_HAND_JOINTS):
            ms = msg.motor_state[i]
            print(fmt_joint(HAND_JOINT_LABELS[i], ms.q, ms.dq))
    except Exception as e:
        print(f"    (could not read motor_state: {e})")

    if not show_pressure:
        return

    press = None
    for attr in ("press_sensor_state", "pressure_state", "press_state"):
        if hasattr(msg, attr):
            press = getattr(msg, attr)
            break

    if press is None:
        print("    (no pressure sensor field on HandState_)")
        return

    print("  pressure modules:")
    try:
        for i in range(min(NUM_PRESS_MODULES, len(press))):
            ps = press[i]
            shown = None
            for attr in ("pressure", "value", "data"):
                if hasattr(ps, attr):
                    shown = getattr(ps, attr)
                    break
            if shown is None:
                shown = ps
            print(f"    sensor[{i}]  {shown}")
    except Exception as e:
        print(f"    (could not read pressure sensors: {e})")


# -------------------------------------------------------------------
# Args + main
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="G1 read-only right-arm + hand state monitor (no commands published)"
    )
    p.add_argument("--iface", default="enx00e04c7015d3",
                   help="Robot ethernet interface (default: enx00e04c7015d3)")
    p.add_argument("--side", choices=["right", "left", "both"], default="right",
                   help="Which Dex3 hand(s) to monitor (default: right)")
    p.add_argument("--no-hand", action="store_true",
                   help="Skip hand monitoring entirely (useful if no Dex3 installed)")
    p.add_argument("--no-pressure", action="store_true",
                   help="Print hand joints but not pressure sensors")
    p.add_argument("--rate", type=float, default=2.0,
                   help="Print rate in Hz (default: 2.0)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-exit after N seconds (default: 0 = until Ctrl-C)")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Seconds to wait for first messages (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print(" G1 RIGHT ARM + HAND STATE MONITOR  (READ-ONLY)")
    print("=" * 64)
    print(f" Interface : {args.iface}")
    print(f" Hand      : {'OFF' if args.no_hand else args.side}")
    print(f" Pressure  : {'OFF' if args.no_pressure or args.no_hand else 'ON'}")
    print(f" Rate      : {args.rate} Hz")
    if args.duration > 0:
        print(f" Duration  : {args.duration} s")
    print("=" * 64)
    print()

    ChannelFactoryInitialize(0, args.iface)

    # Arm subscriber
    arm_mon = LowStateMonitor()
    arm_sub = ChannelSubscriber("rt/lowstate", LowState_)
    arm_sub.Init(arm_mon.handler, 10)
    print("Subscribed to rt/lowstate (arm).")

    # Hand subscribers
    hand_monitors = {}
    if not args.no_hand:
        sides = ["right", "left"] if args.side == "both" else [args.side]
        for side in sides:
            topic = f"rt/dex3/{side}/state"
            mon = HandStateMonitor(side)
            sub = ChannelSubscriber(topic, HandState_)
            sub.Init(mon.handler, 10)
            hand_monitors[side] = (mon, sub, topic)
            print(f"Subscribed to {topic} (hand).")

    print()
    print("Waiting for first messages...")
    t_wait_start = time.time()
    while True:
        arm_ok = arm_mon.first_received
        hand_ok = all(m.first_received for (m, _, _) in hand_monitors.values()) \
                  if hand_monitors else True
        if arm_ok and hand_ok:
            break
        if time.time() - t_wait_start > args.timeout:
            print()
            print(f"Timeout after {args.timeout:.0f}s.")
            print(f"  rt/lowstate         : {'received' if arm_ok else 'NO messages'}")
            for side, (m, _, topic) in hand_monitors.items():
                print(f"  {topic:24s}: {'received' if m.first_received else 'NO messages'}")
            if not arm_ok:
                print("\nArm state not arriving — robot off, wrong iface, or DDS not reaching.")
                print("Run  ip link  to verify the interface name.")
                return
            if hand_monitors and not hand_ok:
                print("\nHand state not arriving — possible reasons:")
                print("  - This robot does not have Dex3 hands (could be Dex1, Inspire, or none).")
                print("  - Hand service on robot's PC2 isn't running.")
                print("  - Cable to hand disconnected.")
                print("Continuing with arm-only display.")
            break
        time.sleep(0.05)

    elapsed = time.time() - t_wait_start
    print(f"First messages received after {elapsed:.2f}s.")
    print()

    # ---- Initial snapshot ----
    print("--- INITIAL SNAPSHOT ---")
    if arm_mon.first_received:
        with arm_mon.lock:
            msg = arm_mon.latest
        print_arm_snapshot(msg)
        print()

    for side, (m, _, _) in hand_monitors.items():
        if not m.first_received:
            print(f"  HAND ({side}):  (no message)")
            continue
        with m.lock:
            msg = m.latest
        print_hand_snapshot(side, msg, show_pressure=not args.no_pressure)
        print()

    print("Sanity checks:")
    print("  Arm:  with arm hanging by side, all q values should be small (|q| < ~0.6).")
    print("  Hand: with hand relaxed, q values reflect resting pose (often a soft fist).")
    print()
    print(f"Streaming at {args.rate} Hz. Ctrl-C to exit.")
    print()

    # ---- Loop ----
    period = 1.0 / max(0.1, args.rate)
    t_start = time.time()
    last_print = 0.0

    try:
        while True:
            now = time.time()

            if now - last_print >= period:
                arm_count = arm_mon.msg_count
                hand_counts = ", ".join(
                    f"{side}={m.msg_count}"
                    for side, (m, _, _) in hand_monitors.items()
                )
                hdr = f"[t={now - t_start:7.2f}s  arm_msgs={arm_count}"
                if hand_counts:
                    hdr += f"  hand_msgs={hand_counts}"
                hdr += "]"
                print(hdr)

                if arm_mon.first_received:
                    with arm_mon.lock:
                        msg = arm_mon.latest
                    print_arm_snapshot(msg)

                for side, (m, _, _) in hand_monitors.items():
                    if not m.first_received:
                        continue
                    with m.lock:
                        msg = m.latest
                    print_hand_snapshot(side, msg, show_pressure=not args.no_pressure)

                print()
                last_print = now

            if args.duration > 0.0 and (now - t_start) >= args.duration:
                print(f"Reached duration limit ({args.duration:.1f}s). Exiting.")
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print()
        print("Interrupted.")

    print()
    print(f"Total arm messages received: {arm_mon.msg_count}")
    for side, (m, _, _) in hand_monitors.items():
        print(f"Total {side}-hand messages received: {m.msg_count}")
    print("Done. No commands were published at any point.")


if __name__ == "__main__":
    main()