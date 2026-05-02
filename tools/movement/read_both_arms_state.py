#!/usr/bin/env python3
"""
G1 BOTH arms read-only diagnostic.

Prints left-arm and right-arm joint states side-by-side from rt/lowstate.
Use this to verify which physical arm corresponds to which joint indices.

Intended location:
    ~/ros2_ws/tools/movement/read_both_arms_state.py

Usage:
    python3 ~/ros2_ws/tools/movement/read_both_arms_state.py
    python3 ~/ros2_ws/tools/movement/read_both_arms_state.py --rate 5

Read-only. No commands published.
"""

import os
import sys
import site

HOME = os.path.expanduser("~")
UNITREE_REPO = os.path.join(HOME, "unitree_sdk2_python")
VENV_SITE = os.path.join(
    HOME, "unitree_sdk2_venv", "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(UNITREE_REPO) and UNITREE_REPO not in sys.path:
    sys.path.insert(0, UNITREE_REPO)
if os.path.isdir(VENV_SITE):
    site.addsitedir(VENV_SITE)

import argparse
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


# Standard G1 29-DOF joint indices (per Unitree docs / common SDK convention)
class G1JointIndex(IntEnum):
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


LEFT_ARM = [
    ("LEFT  shoulder_pitch", G1JointIndex.LeftShoulderPitch),
    ("LEFT  shoulder_roll ", G1JointIndex.LeftShoulderRoll),
    ("LEFT  shoulder_yaw  ", G1JointIndex.LeftShoulderYaw),
    ("LEFT  elbow         ", G1JointIndex.LeftElbow),
    ("LEFT  wrist_roll    ", G1JointIndex.LeftWristRoll),
    ("LEFT  wrist_pitch   ", G1JointIndex.LeftWristPitch),
    ("LEFT  wrist_yaw     ", G1JointIndex.LeftWristYaw),
]

RIGHT_ARM = [
    ("RIGHT shoulder_pitch", G1JointIndex.RightShoulderPitch),
    ("RIGHT shoulder_roll ", G1JointIndex.RightShoulderRoll),
    ("RIGHT shoulder_yaw  ", G1JointIndex.RightShoulderYaw),
    ("RIGHT elbow         ", G1JointIndex.RightElbow),
    ("RIGHT wrist_roll    ", G1JointIndex.RightWristRoll),
    ("RIGHT wrist_pitch   ", G1JointIndex.RightWristPitch),
    ("RIGHT wrist_yaw     ", G1JointIndex.RightWristYaw),
]


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.first = False
        self.count = 0

    def handler(self, msg: LowState_):
        with self.lock:
            self.latest = msg
            self.count += 1
            self.first = True


def fmt_row(label, idx, msg):
    ms = msg.motor_state[idx]
    return f"  [{idx:2d}]  {label}  q = {ms.q:+8.4f} rad   dq = {ms.dq:+8.4f} rad/s"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="enx00e04c7015d3")
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print(" G1 BOTH ARMS STATE  (READ-ONLY)")
    print("=" * 64)
    print(" Wiggle ONE arm at a time by hand and watch which indices move.")
    print(" Compare to label to verify left/right indexing is correct.")
    print("=" * 64)
    print()

    ChannelFactoryInitialize(0, args.iface)
    mon = Monitor()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(mon.handler, 10)

    print("Waiting for first lowstate...")
    t0 = time.time()
    while not mon.first:
        if time.time() - t0 > 10:
            print("No lowstate received. Robot off?")
            return
        time.sleep(0.05)
    print(f"Received after {time.time() - t0:.2f}s.\n")

    period = 1.0 / max(0.1, args.rate)
    last = 0.0
    t_start = time.time()

    try:
        while True:
            now = time.time()
            if now - last >= period:
                with mon.lock:
                    msg = mon.latest
                    count = mon.count
                print(f"[t={now - t_start:6.2f}s msgs={count}]")
                for label, idx in LEFT_ARM:
                    print(fmt_row(label, idx, msg))
                print()
                for label, idx in RIGHT_ARM:
                    print(fmt_row(label, idx, msg))
                print()
                last = now

            if args.duration > 0 and (now - t_start) >= args.duration:
                break

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nTotal messages: {mon.count}. No commands were published.")


if __name__ == "__main__":
    main()