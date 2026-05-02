#!/usr/bin/env python3
"""
G1 single-arm-motor identification test.

Diagnoses whether the previous "wrong arm moved" issue was:
  (a) a real cmd-side index mismatch, OR
  (b) the previous script not writing to non-target arm joints, which
      caused those joints to be commanded to default values (q=0)
      when arm_sdk weight snapped to 1.0.

This script:
  - Writes to ALL 14 arm joints (left arm 15-21, right arm 22-28).
  - Pins 13 of them to their CURRENT q readings (gentle hold, low kp).
  - Applies a small delta to ONE configurable index.
  - Ramps the arm_sdk weight 0 -> 1 over 1s (no snap).
  - Holds 2s, then ramps weight 1 -> 0.

Run the both-arms monitor in another terminal at the same time:
    python3 ~/ros2_ws/tools/movement/read_both_arms_state.py --rate 10

Usage:
    # Default: nudge index 25 (right_elbow per state-side mapping)
    python3 ~/ros2_ws/tools/movement/test_single_arm_motor.py

    # Test a different index
    python3 ~/ros2_ws/tools/movement/test_single_arm_motor.py --index 18

    # Custom delta (radians)
    python3 ~/ros2_ws/tools/movement/test_single_arm_motor.py --delta -0.05
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

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# Standard G1 29-DOF mapping (state side, confirmed by Caleb's diagnostic)
ARM_INDICES = {
    15: "LEFT  shoulder_pitch",
    16: "LEFT  shoulder_roll",
    17: "LEFT  shoulder_yaw",
    18: "LEFT  elbow",
    19: "LEFT  wrist_roll",
    20: "LEFT  wrist_pitch",
    21: "LEFT  wrist_yaw",
    22: "RIGHT shoulder_pitch",
    23: "RIGHT shoulder_roll",
    24: "RIGHT shoulder_yaw",
    25: "RIGHT elbow",
    26: "RIGHT wrist_roll",
    27: "RIGHT wrist_pitch",
    28: "RIGHT wrist_yaw",
}
KNOT_USED_JOINT = 29  # arm_sdk weight slot


class Controller:
    def __init__(self, args):
        self.iface = args.iface
        self.target_index = args.index
        self.delta = args.delta
        self.kp_hold = args.kp_hold
        self.kp_test = args.kp_test
        self.kd = args.kd
        self.weight_ramp_sec = args.weight_ramp
        self.move_sec = args.duration
        self.hold_sec = args.hold
        self.release_sec = args.release

        self.control_dt = 0.02
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.lock = threading.Lock()

        # Per-index start_q and target_q for all 14 arm joints
        self.start_q = {idx: 0.0 for idx in ARM_INDICES}
        self.target_q = {idx: 0.0 for idx in ARM_INDICES}

        self.stage = "wait"
        self.t_stage = 0.0
        self.done = False

        self.pub = None
        self.sub = None
        self.thread = None

    def init_channels(self):
        ChannelFactoryInitialize(0, self.iface)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            if not self.first_state:
                self.first_state = True
                # Snapshot all 14 arm joints' current q
                for idx in ARM_INDICES:
                    self.start_q[idx] = msg.motor_state[idx].q
                # Build target_q: every joint stays at current EXCEPT target_index
                for idx in ARM_INDICES:
                    self.target_q[idx] = self.start_q[idx]
                if self.target_index in ARM_INDICES:
                    self.target_q[self.target_index] = (
                        self.start_q[self.target_index] + self.delta
                    )

    def print_plan(self):
        print()
        print("=" * 72)
        print(f" Target index : {self.target_index}  ({ARM_INDICES.get(self.target_index, '???')})")
        print(f" Delta        : {self.delta:+.4f} rad")
        print(f" kp_hold      : {self.kp_hold}   (for the 13 pinned joints)")
        print(f" kp_test      : {self.kp_test}   (for the target joint)")
        print(f" kd           : {self.kd}")
        print(f" Weight ramp  : 0 -> 1 over {self.weight_ramp_sec}s")
        print(f" Move ramp    : start_q -> target_q over {self.move_sec}s")
        print(f" Hold         : {self.hold_sec}s")
        print(f" Release      : weight 1 -> 0 over {self.release_sec}s")
        print("=" * 72)
        print(f"{'idx':>4}  {'name':<24}  {'start_q':>10}  {'target_q':>10}  {'delta':>9}")
        for idx in sorted(ARM_INDICES):
            s = self.start_q[idx]
            t = self.target_q[idx]
            d = t - s
            mark = "  <-- TARGET" if idx == self.target_index else ""
            print(f"{idx:>4}  {ARM_INDICES[idx]:<24}  {s:+10.4f}  {t:+10.4f}  {d:+9.4f}{mark}")
        print("=" * 72)

    def _set_motor(self, idx, q, kp, kd):
        m = self.low_cmd.motor_cmd[idx]
        m.mode = 1
        m.q = q
        m.dq = 0.0
        m.tau = 0.0
        m.kp = kp
        m.kd = kd

    def _write_arm_joints(self, blend_ratio):
        """Write all 14 arm joints. blend_ratio in [0,1] from start_q to target_q for the test joint."""
        for idx in ARM_INDICES:
            if idx == self.target_index:
                q = self.start_q[idx] + (self.target_q[idx] - self.start_q[idx]) * blend_ratio
                self._set_motor(idx, q, self.kp_test, self.kd)
            else:
                # Pin to current
                self._set_motor(idx, self.start_q[idx], self.kp_hold, self.kd)

    def _publish(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def start(self):
        print("Waiting for rt/lowstate ...")
        while not self.first_state:
            time.sleep(0.05)

        self.print_plan()
        input("\nReview the plan. Press Enter to begin (Ctrl-C aborts)... ")

        self.stage = "weight_up"
        self.t_stage = time.time()

        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self._tick,
            name="single_arm_motor_test",
        )
        self.thread.Start()

    def _tick(self):
        if self.done:
            return
        now = time.time()

        if self.stage == "weight_up":
            r = max(0.0, min(1.0, (now - self.t_stage) / self.weight_ramp_sec))
            self.low_cmd.motor_cmd[KNOT_USED_JOINT].q = r
            self._write_arm_joints(blend_ratio=0.0)  # all at start_q
            self._publish()
            if r >= 1.0:
                self.stage = "move"
                self.t_stage = now
                print("[stage] weight=1.0 reached, beginning move")

        elif self.stage == "move":
            r = max(0.0, min(1.0, (now - self.t_stage) / self.move_sec))
            self.low_cmd.motor_cmd[KNOT_USED_JOINT].q = 1.0
            self._write_arm_joints(blend_ratio=r)
            self._publish()
            if r >= 1.0:
                self.stage = "hold"
                self.t_stage = now
                print("[stage] target reached, holding")

        elif self.stage == "hold":
            self.low_cmd.motor_cmd[KNOT_USED_JOINT].q = 1.0
            self._write_arm_joints(blend_ratio=1.0)
            self._publish()
            if (now - self.t_stage) >= self.hold_sec:
                self.stage = "release"
                self.t_stage = now
                print("[stage] hold done, ramping weight to 0")

        elif self.stage == "release":
            r = max(0.0, min(1.0, (now - self.t_stage) / self.release_sec))
            self.low_cmd.motor_cmd[KNOT_USED_JOINT].q = 1.0 - r
            self._write_arm_joints(blend_ratio=1.0)
            self._publish()
            if r >= 1.0:
                print("[stage] done, arm_sdk released")
                self.done = True


def parse_args():
    p = argparse.ArgumentParser(
        description="G1 single-arm-motor identification test"
    )
    p.add_argument("--iface", default="enx00e04c7015d3")
    p.add_argument("--index", type=int, default=25,
                   help="Motor index to nudge (default 25 = right elbow per state mapping)")
    p.add_argument("--delta", type=float, default=-0.05,
                   help="Position delta in radians (default -0.05, negative=extend if elbow)")
    p.add_argument("--kp-hold", type=float, default=10.0,
                   help="kp for the 13 pinned joints (default 10)")
    p.add_argument("--kp-test", type=float, default=15.0,
                   help="kp for the target joint (default 15)")
    p.add_argument("--kd", type=float, default=1.5)
    p.add_argument("--weight-ramp", type=float, default=1.5,
                   help="Seconds to ramp arm_sdk weight 0->1 (default 1.5)")
    p.add_argument("--duration", type=float, default=3.0,
                   help="Seconds for the single-joint move (default 3)")
    p.add_argument("--hold", type=float, default=2.0)
    p.add_argument("--release", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.index not in ARM_INDICES:
        print(f"ERROR: --index {args.index} is not an arm joint.")
        print("Valid indices:", sorted(ARM_INDICES.keys()))
        sys.exit(1)

    print("=" * 72)
    print(" G1 SINGLE-ARM-MOTOR IDENTIFICATION TEST")
    print("=" * 72)
    print(" This will:")
    print("  1. Read current pose for all 14 arm joints (15-21 left, 22-28 right)")
    print("  2. Pin 13 joints to their current q (low kp, gentle hold)")
    print("  3. Move ONE joint by a small delta")
    print("  4. Ramp arm_sdk weight 0 -> 1 (no snap), hold, then release")
    print()
    print(" RUN THIS IN ANOTHER TERMINAL FIRST:")
    print("   python3 ~/ros2_ws/tools/movement/read_both_arms_state.py --rate 10")
    print()
    print(" Robot should be in motion control mode (Regular Mode).")
    print(" Hand near L2+B for emergency damping.")
    print("=" * 72)

    ctrl = Controller(args)
    ctrl.init_channels()
    ctrl.start()

    try:
        while not ctrl.done:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted. Note: weight may still be 1.0 -- press L2+B for safety.")


if __name__ == "__main__":
    main()