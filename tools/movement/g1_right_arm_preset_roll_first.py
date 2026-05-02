#!/usr/bin/env python3
"""
G1 right-arm preset move — safer roll-first version for real robot.

Main safety rules preserved:
1. Writes ALL 14 arm joints, not only the right arm.
2. Pins left arm to its current pose.
3. Ramps arm_sdk weight 0 -> 1 gradually.
4. Computes CRC before every LowCmd publish.
5. Asks for Enter after printing exact start/target deltas.

New for avoiding right-leg / diver collision:
- ready_small_out: same basic ready pose, but right_shoulder_roll is more negative
  so the arm moves slightly outward before elbow extension.
- --roll-first: phases the move as:
    phase 1: right shoulder roll outward
    phase 2: shoulder pitch/yaw
    phase 3: elbow/wrist extension
- --roll-only: safest first test. Only right_shoulder_roll moves; everything else
  on both arms is pinned to current measured q.

Recommended first test:
  python3 ~/ros2_ws/tools/movement/g1_right_arm_preset.py \
    --preset ready_small_out --roll-only --right-shoulder-roll -0.18 \
    --duration 3.0 --hold 0.5 --release 2.0

Recommended second test:
  python3 ~/ros2_ws/tools/movement/g1_right_arm_preset.py \
    --preset ready_small_out --roll-first --gradual-step 0.35 \
    --duration 6.0 --hold 0.5 --release 2.0

Only after that:
  python3 ~/ros2_ws/tools/movement/g1_right_arm_preset.py \
    --preset ready_small_out --roll-first --gradual-step 1.0 \
    --duration 7.0 --hold 1.0 --release 2.0

Robot must be suspended, in Regular Mode, with hand near L2+B damping.
"""

import os
import sys
import site

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

import argparse
import threading
import time
from typing import Dict, List

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29


LEFT_ARM_INDICES = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

RIGHT_ARM_INDICES = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

RIGHT_ARM_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

RIGHT_SHOULDER_PITCH_I = 0
RIGHT_SHOULDER_ROLL_I = 1
RIGHT_SHOULDER_YAW_I = 2
RIGHT_ELBOW_I = 3

SHOULDER_JOINT_MASK = [True, True, True, False, False, False, False]


PRESETS: Dict[str, Dict[str, float]] = {
    # Original-ish pose, but with shoulder_roll made more negative than the
    # measured rest value (~-0.136). This should move the arm slightly outward.
    "ready_small_out": {
        "right_shoulder_pitch": 0.15,
        "right_shoulder_roll": -0.20,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.35,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
    },
    # Kept for comparison / fallback.
    "ready_small": {
        "right_shoulder_pitch": 0.15,
        "right_shoulder_roll": -0.10,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.35,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
    },
    "ready_sim": {
        "right_shoulder_pitch": 0.10,
        "right_shoulder_roll": -0.20,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.20,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
    },
    "reach_small": {
        "right_shoulder_pitch": 0.25,
        "right_shoulder_roll": -0.15,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.55,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": -0.10,
        "right_wrist_yaw": 0.0,
    },
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ramp_between(t_norm: float, start: float, end: float) -> float:
    """Linear 0->1 blend over [start, end]."""
    if end <= start:
        return 1.0 if t_norm >= end else 0.0
    return clamp01((t_norm - start) / (end - start))


class Controller:
    def __init__(self, args):
        self.iface = args.iface
        self.preset = args.preset
        self.gradual_step = clamp01(args.gradual_step)
        self.lift_first = args.lift_first
        self.roll_first = args.roll_first
        self.roll_only = args.roll_only
        self.kp_test = args.kp_test
        self.kp_hold = args.kp_hold
        self.kd = args.kd
        self.weight_ramp_sec = args.weight_ramp
        self.move_sec = args.duration
        self.hold_sec = args.hold
        self.release_sec = args.release
        self.overrides = args.overrides
        self.roll_done_fraction = args.roll_done_fraction
        self.shoulder_start_fraction = args.shoulder_start_fraction
        self.shoulder_done_fraction = args.shoulder_done_fraction
        self.elbow_start_fraction = args.elbow_start_fraction

        self.control_dt = 0.02
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.lock = threading.Lock()

        self.left_start_q = [0.0] * 7
        self.right_start_q = [0.0] * 7
        self.right_target_q = [0.0] * 7

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
                self.left_start_q = [msg.motor_state[i].q for i in LEFT_ARM_INDICES]
                self.right_start_q = [msg.motor_state[i].q for i in RIGHT_ARM_INDICES]

                preset_q = [PRESETS[self.preset][name] for name in RIGHT_ARM_NAMES]
                for i, name in enumerate(RIGHT_ARM_NAMES):
                    if name in self.overrides:
                        preset_q[i] = self.overrides[name]

                if self.roll_only:
                    # Safest possible test: keep every right-arm joint at the measured
                    # start pose except shoulder_roll.
                    target = list(self.right_start_q)
                    target[RIGHT_SHOULDER_ROLL_I] = preset_q[RIGHT_SHOULDER_ROLL_I]
                    preset_q = target

                self.right_target_q = [
                    self.right_start_q[i]
                    + self.gradual_step * (preset_q[i] - self.right_start_q[i])
                    for i in range(7)
                ]

    def print_plan(self):
        print()
        print("=" * 80)
        print(f" Preset       : {self.preset}")
        print(f" Gradual step : {self.gradual_step:.2f}  ({int(self.gradual_step * 100)}% of way to preset)")
        print(f" Roll only    : {self.roll_only}")
        print(f" Roll first   : {self.roll_first}")
        print(f" Lift first   : {self.lift_first}")
        print(f" kp_test      : {self.kp_test} (right arm)")
        print(f" kp_hold      : {self.kp_hold} (left arm pinned)")
        print(f" kd           : {self.kd}")
        print(f" Weight ramp  : 0 -> 1 over {self.weight_ramp_sec}s")
        print(f" Move duration: {self.move_sec}s")
        if self.roll_first:
            print(" Motion phases:")
            print(f"   right_shoulder_roll : 0.00 - {self.roll_done_fraction:.2f} of move")
            print(f"   shoulder pitch/yaw  : {self.shoulder_start_fraction:.2f} - {self.shoulder_done_fraction:.2f} of move")
            print(f"   elbow/wrist         : {self.elbow_start_fraction:.2f} - 1.00 of move")
        elif self.lift_first:
            print(" Motion phases:")
            print("   shoulder joints     : 0.00 - 0.60 of move")
            print("   elbow/wrist         : 0.30 - 1.00 of move")
        print(f" Hold         : {self.hold_sec}s")
        print(f" Release      : weight 1 -> 0 over {self.release_sec}s")
        print("=" * 80)
        print(f"{'idx':>4}  {'name':<24}  {'start_q':>10}  {'target_q':>10}  {'delta':>9}")
        print(f"  {'-- LEFT ARM (pinned to current) --':<24}")
        for i, idx in enumerate(LEFT_ARM_INDICES):
            print(f"{idx:>4}  LEFT  joint {i:<13}  {self.left_start_q[i]:+10.4f}  {self.left_start_q[i]:+10.4f}  {0.0:+9.4f}")
        print(f"  {'-- RIGHT ARM (commanded) --':<24}")
        for i, idx in enumerate(RIGHT_ARM_INDICES):
            s = self.right_start_q[i]
            t = self.right_target_q[i]
            d = t - s
            mark = ""
            if i == RIGHT_SHOULDER_ROLL_I:
                mark = "  <-- SHOULDER ROLL / clearance"
            elif i == RIGHT_ELBOW_I:
                mark = "  <-- elbow extension"
            print(f"{idx:>4}  RIGHT {RIGHT_ARM_NAMES[i]:<18}  {s:+10.4f}  {t:+10.4f}  {d:+9.4f}{mark}")
        print("=" * 80)

        roll_delta = self.right_target_q[RIGHT_SHOULDER_ROLL_I] - self.right_start_q[RIGHT_SHOULDER_ROLL_I]
        if roll_delta > 0.0:
            print("WARNING: right_shoulder_roll target is MORE positive than start.")
            print("         Based on your previous test, more negative is expected to move outward.")
            print("         Abort with Ctrl-C if this looks wrong.")
        else:
            print(f"Right shoulder roll will move more negative by {abs(roll_delta):.4f} rad ({abs(roll_delta) * 57.3:.1f} deg).")

        max_d = max(abs(self.right_target_q[i] - self.right_start_q[i]) for i in range(7))
        print(f"Largest single-joint motion: {max_d:.4f} rad ({max_d * 57.3:.1f} deg)")
        print("=" * 80)

    def _set_motor(self, idx: int, q: float, kp: float, kd: float):
        m = self.low_cmd.motor_cmd[idx]
        m.mode = 1
        m.q = q
        m.dq = 0.0
        m.tau = 0.0
        m.kp = kp
        m.kd = kd

    def _compute_joint_blends(self, t_norm: float) -> List[float]:
        """Return seven right-arm blend values, one per right-arm joint."""
        t_norm = clamp01(t_norm)

        if self.roll_only:
            blends = [0.0] * 7
            blends[RIGHT_SHOULDER_ROLL_I] = t_norm
            return blends

        if self.roll_first:
            blends = [0.0] * 7

            # 1) Roll outward first to clear fingers/forearm from thigh area.
            blends[RIGHT_SHOULDER_ROLL_I] = ramp_between(
                t_norm, 0.0, self.roll_done_fraction
            )

            # 2) Other shoulder joints after roll is underway.
            shoulder_blend = ramp_between(
                t_norm, self.shoulder_start_fraction, self.shoulder_done_fraction
            )
            blends[RIGHT_SHOULDER_PITCH_I] = shoulder_blend
            blends[RIGHT_SHOULDER_YAW_I] = shoulder_blend

            # 3) Only then allow elbow/wrist to move.
            elbow_blend = ramp_between(t_norm, self.elbow_start_fraction, 1.0)
            for i in range(3, 7):
                blends[i] = elbow_blend
            return blends

        if self.lift_first:
            shoulder_b = min(1.0, t_norm / 0.6)
            elbow_b = 0.0 if t_norm <= 0.3 else min(1.0, (t_norm - 0.3) / 0.7)
            return [shoulder_b if SHOULDER_JOINT_MASK[i] else elbow_b for i in range(7)]

        return [t_norm] * 7

    def _write_arm_joints(self, right_blends: List[float]):
        # Pin left arm to current measured pose.
        for i, idx in enumerate(LEFT_ARM_INDICES):
            self._set_motor(idx, self.left_start_q[i], self.kp_hold, self.kd)

        # Command right arm with per-joint phasing.
        for i, idx in enumerate(RIGHT_ARM_INDICES):
            blend = right_blends[i]
            q = self.right_start_q[i] + (self.right_target_q[i] - self.right_start_q[i]) * blend
            self._set_motor(idx, q, self.kp_test, self.kd)

    def _publish(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def release_arm_sdk_now(self):
        """Best-effort soft release if Ctrl-C is pressed."""
        try:
            self.done = True
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0.0
            self._write_arm_joints([0.0] * 7)
            for _ in range(10):
                self._publish()
                time.sleep(0.02)
            print("[safety] arm_sdk weight commanded to 0. Press L2+B if anything looks wrong.")
        except Exception as exc:
            print(f"[safety] Could not publish release command: {exc}")
            print("[safety] Press L2+B for damping.")

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
            name="g1_right_arm_preset_roll_first",
        )
        self.thread.Start()

    def _tick(self):
        if self.done:
            return
        now = time.time()

        if self.stage == "weight_up":
            r = clamp01((now - self.t_stage) / self.weight_ramp_sec)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = r
            self._write_arm_joints([0.0] * 7)
            self._publish()
            if r >= 1.0:
                self.stage = "move"
                self.t_stage = now
                print("[stage] weight=1.0 reached, moving")

        elif self.stage == "move":
            t_norm = clamp01((now - self.t_stage) / self.move_sec)
            blends = self._compute_joint_blends(t_norm)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
            self._write_arm_joints(blends)
            self._publish()
            if t_norm >= 1.0:
                self.stage = "hold"
                self.t_stage = now
                print("[stage] target reached, holding")

        elif self.stage == "hold":
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
            self._write_arm_joints([1.0] * 7)
            self._publish()
            if (now - self.t_stage) >= self.hold_sec:
                self.stage = "release"
                self.t_stage = now
                print("[stage] hold done, ramping weight to 0")

        elif self.stage == "release":
            r = clamp01((now - self.t_stage) / self.release_sec)
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - r
            self._write_arm_joints([1.0] * 7)
            self._publish()
            if r >= 1.0:
                print("[stage] done, arm_sdk released")
                self.done = True


def parse_args():
    p = argparse.ArgumentParser(
        description="G1 right-arm preset move with safer shoulder-roll-first option."
    )
    p.add_argument("--iface", default="enx00e04c7015d3")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="ready_small_out")
    p.add_argument(
        "--gradual-step",
        type=float,
        default=1.0,
        help="Fraction of the way from current toward preset (0.0-1.0).",
    )
    p.add_argument(
        "--lift-first",
        action="store_true",
        help="Old behaviour: shoulder joints lead, elbow/wrist follow.",
    )
    p.add_argument(
        "--roll-first",
        action="store_true",
        help="New safer behaviour: shoulder_roll moves first, then shoulder pitch/yaw, then elbow/wrist.",
    )
    p.add_argument(
        "--roll-only",
        action="store_true",
        help="Safest test: only right_shoulder_roll moves; every other arm joint is pinned to current q.",
    )
    p.add_argument("--roll-done-fraction", type=float, default=0.35)
    p.add_argument("--shoulder-start-fraction", type=float, default=0.15)
    p.add_argument("--shoulder-done-fraction", type=float, default=0.70)
    p.add_argument("--elbow-start-fraction", type=float, default=0.60)

    p.add_argument("--kp-test", type=float, default=35.0, help="kp for right arm. Lower than before for first real tests.")
    p.add_argument("--kp-hold", type=float, default=40.0, help="kp for pinned left arm.")
    p.add_argument("--kd", type=float, default=1.5)
    p.add_argument("--weight-ramp", type=float, default=1.5)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--hold", type=float, default=0.5)
    p.add_argument("--release", type=float, default=2.0)

    p.add_argument("--right-shoulder-pitch", type=float, default=None)
    p.add_argument("--right-shoulder-roll", type=float, default=None)
    p.add_argument("--right-shoulder-yaw", type=float, default=None)
    p.add_argument("--right-elbow", type=float, default=None)
    p.add_argument("--right-wrist-roll", type=float, default=None)
    p.add_argument("--right-wrist-pitch", type=float, default=None)
    p.add_argument("--right-wrist-yaw", type=float, default=None)

    args = p.parse_args()

    if args.roll_only:
        args.roll_first = False
        args.lift_first = False
    if args.roll_first:
        args.lift_first = False

    # Keep phase fractions sane even if mistyped.
    args.roll_done_fraction = clamp01(args.roll_done_fraction)
    args.shoulder_start_fraction = clamp01(args.shoulder_start_fraction)
    args.shoulder_done_fraction = clamp01(args.shoulder_done_fraction)
    args.elbow_start_fraction = clamp01(args.elbow_start_fraction)

    overrides: Dict[str, float] = {}
    arg_map = {
        "right_shoulder_pitch": args.right_shoulder_pitch,
        "right_shoulder_roll": args.right_shoulder_roll,
        "right_shoulder_yaw": args.right_shoulder_yaw,
        "right_elbow": args.right_elbow,
        "right_wrist_roll": args.right_wrist_roll,
        "right_wrist_pitch": args.right_wrist_pitch,
        "right_wrist_yaw": args.right_wrist_yaw,
    }
    for k, v in arg_map.items():
        if v is not None:
            overrides[k] = float(v)
    args.overrides = overrides
    return args


def main():
    args = parse_args()
    print("=" * 80)
    print(" G1 RIGHT-ARM PRESET MOVE — ROLL-FIRST SAFER VERSION")
    print("=" * 80)
    print("Robot must be suspended and in Regular Mode: L2+UP, then R1+X.")
    print("Keep hand on L2+B for emergency damping.")
    print("Avoid L2+R2 Debug Mode.")
    print("Stand outside the arm sweep path.")
    print("=" * 80)

    ctrl = Controller(args)
    ctrl.init_channels()
    try:
        ctrl.start()
        while not ctrl.done:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl-C.")
        ctrl.release_arm_sdk_now()


if __name__ == "__main__":
    main()
