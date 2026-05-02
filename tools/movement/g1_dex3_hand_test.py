#!/usr/bin/env python3
"""
G1 Dex3 right hand open/close test with pressure-gated grasp.

This script tests hand control independently from arm motion.
It publishes HandCmd_ to rt/dex3/right/cmd and monitors pressure
via HandState_ on rt/dex3/right/state.

Flow:
  1. Subscribe to hand state, read current q + pressure baseline
  2. Ramp to OPEN pose over open_duration
  3. Hold open
  4. Gradually ramp toward CLOSED pose
  5. Monitor pressure each tick — if any sensor exceeds baseline + threshold,
     freeze ALL fingers at current position
  6. Hold gripped pose
  7. Optionally release (ramp back to OPEN)

Usage:
  # Open then close with pressure stop
  python3 g1_dex3_hand_test.py

  # Open only (test hand opening)
  python3 g1_dex3_hand_test.py --open-only

  # Adjust pressure threshold
  python3 g1_dex3_hand_test.py --pressure-threshold 2000

Safety: hand should be in Regular Mode (R1+X). No arm motion.
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
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_


# ── Hand poses (measured from real Dex3-1) ────────────────────────────

HAND_JOINT_NAMES = [
    "thumb_abduction",
    "thumb_flex_0",
    "thumb_flex_1",
    "finger_left_0",
    "finger_left_1",
    "finger_right_0",
    "finger_right_1",
]

OPEN_HAND = [+0.012, +0.577, -0.029, -0.085, -0.069, -0.017, -0.082]
CLOSED_HAND = [+0.012, -1.009, -1.650, +1.488, +1.671, +1.493, +1.686]

NUM_JOINTS = 7
NUM_SENSORS = 9
INACTIVE_THRESHOLD = 50000.0  # sensors reading 30000 are inactive


class HandController:
    def __init__(self, args):
        self.iface = args.iface
        self.open_only = args.open_only
        self.open_duration = args.open_duration
        self.open_hold = args.open_hold
        self.close_duration = args.close_duration
        self.close_hold = args.close_hold
        self.release_duration = args.release_duration
        self.kp = args.hand_kp
        self.kd = args.hand_kd
        self.pressure_threshold = args.pressure_threshold

        self.control_dt = 0.02
        self.lock = threading.Lock()

        self.hand_state = None
        self.first_state = False
        self.start_q = [0.0] * NUM_JOINTS
        self.pressure_baseline = None  # captured when hand is open

        # Current command position (updated during close if pressure trips)
        self.frozen = False
        self.frozen_q = None

        self.stage = "wait"
        self.t_stage = 0.0
        self.done = False

    def init_channels(self):
        ChannelFactoryInitialize(0, self.iface)

        self.pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg: HandState_):
        with self.lock:
            self.hand_state = msg
            if not self.first_state:
                self.first_state = True
                self.start_q = [msg.motor_state[i].q for i in range(NUM_JOINTS)]

    def _get_pressure_readings(self):
        """Extract all active pressure values from current hand state."""
        with self.lock:
            if self.hand_state is None:
                return []
            msg = self.hand_state

        readings = []
        press = None
        for attr in ("press_sensor_state", "pressure_state", "press_state"):
            if hasattr(msg, attr):
                press = getattr(msg, attr)
                break

        if press is None:
            return []

        try:
            for i in range(min(NUM_SENSORS, len(press))):
                ps = press[i]
                # Each sensor module has multiple values (12 per module)
                vals = None
                for attr in ("pressure", "value", "data"):
                    if hasattr(ps, attr):
                        vals = getattr(ps, attr)
                        break
                if vals is None:
                    continue
                if hasattr(vals, '__iter__'):
                    for v in vals:
                        if v > INACTIVE_THRESHOLD:  # skip 30000 inactive channels
                            readings.append(v)
                else:
                    if vals > INACTIVE_THRESHOLD:
                        readings.append(vals)
        except Exception:
            pass

        return readings

    def _capture_pressure_baseline(self):
        """Capture current pressure as baseline (called when hand is open)."""
        readings = self._get_pressure_readings()
        if readings:
            self.pressure_baseline = readings
            print(f"  Pressure baseline captured: {len(readings)} active channels")
            print(f"  Baseline range: {min(readings):.0f} - {max(readings):.0f}")
        else:
            print("  WARNING: No pressure readings available. Closing will be unguarded.")
            self.pressure_baseline = None

    def _check_pressure_exceeded(self):
        """Check if any pressure sensor exceeds baseline + threshold."""
        if self.pressure_baseline is None:
            return False

        current = self._get_pressure_readings()
        if len(current) != len(self.pressure_baseline):
            return False

        for i in range(len(current)):
            delta = current[i] - self.pressure_baseline[i]
            if delta > self.pressure_threshold:
                return True
        return False

    def _write_hand(self, q_7):
        """Publish hand command."""
        cmd = HandCmd_()
        for i in range(NUM_JOINTS):
            cmd.motor_cmd[i].q = q_7[i]
            cmd.motor_cmd[i].kp = self.kp
            cmd.motor_cmd[i].kd = self.kd
            cmd.motor_cmd[i].mode = 1
        self.pub.Write(cmd)

    def _blend(self, start, end, t_norm):
        t = max(0.0, min(1.0, t_norm))
        return [start[i] + t * (end[i] - start[i]) for i in range(NUM_JOINTS)]

    def start(self):
        print("Waiting for rt/dex3/right/state ...")
        t0 = time.time()
        while not self.first_state:
            if time.time() - t0 > 10:
                print("No hand state received. Is the Dex3 hand connected?")
                return False
            time.sleep(0.05)

        print(f"Hand state received. Current q:")
        for i, name in enumerate(HAND_JOINT_NAMES):
            print(f"  {name:<20} = {self.start_q[i]:+.4f}")

        print(f"\nTarget OPEN:   {[f'{v:+.3f}' for v in OPEN_HAND]}")
        print(f"Target CLOSED: {[f'{v:+.3f}' for v in CLOSED_HAND]}")
        print(f"Pressure threshold: {self.pressure_threshold}")

        input("\nPress Enter to begin (Ctrl-C aborts)... ")
        return True

    def run(self):
        self.stage = "open"
        self.t_stage = time.time()
        print("[stage] opening hand")

        try:
            while not self.done:
                self._tick()
                time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("\nInterrupted.")

    def _tick(self):
        if self.done:
            return
        now = time.time()

        if self.stage == "open":
            t_norm = (now - self.t_stage) / self.open_duration
            t_norm = max(0.0, min(1.0, t_norm))
            q = self._blend(self.start_q, OPEN_HAND, t_norm)
            self._write_hand(q)
            if t_norm >= 1.0:
                self.stage = "hold_open"
                self.t_stage = now
                self._capture_pressure_baseline()
                print("[stage] hand open, holding")

        elif self.stage == "hold_open":
            self._write_hand(OPEN_HAND)
            if (now - self.t_stage) >= self.open_hold:
                if self.open_only:
                    self.stage = "release"
                    self.t_stage = now
                    print("[stage] open-only, releasing")
                else:
                    self.stage = "close"
                    self.t_stage = now
                    self.frozen = False
                    print("[stage] closing hand (pressure-gated)")

        elif self.stage == "close":
            if not self.frozen:
                t_norm = (now - self.t_stage) / self.close_duration
                t_norm = max(0.0, min(1.0, t_norm))
                q = self._blend(OPEN_HAND, CLOSED_HAND, t_norm)

                # Check pressure
                if self._check_pressure_exceeded():
                    self.frozen = True
                    self.frozen_q = list(q)
                    print(f"[PRESSURE] Contact detected at t_norm={t_norm:.2f}! Freezing fingers.")
                    for i, name in enumerate(HAND_JOINT_NAMES):
                        print(f"  {name:<20} frozen at {q[i]:+.4f}")
                else:
                    self._write_hand(q)

                if t_norm >= 1.0 and not self.frozen:
                    self.frozen = True
                    self.frozen_q = list(CLOSED_HAND)
                    print("[stage] fully closed (no pressure trigger)")
            else:
                self._write_hand(self.frozen_q)

            if self.frozen:
                if not hasattr(self, '_hold_start'):
                    self._hold_start = now
                if (now - self._hold_start) >= self.close_hold:
                    self.stage = "release"
                    self.t_stage = now
                    print("[stage] hold done, releasing")

        elif self.stage == "release":
            t_norm = (now - self.t_stage) / self.release_duration
            t_norm = max(0.0, min(1.0, t_norm))
            release_from = self.frozen_q if self.frozen_q else OPEN_HAND
            q = self._blend(release_from, OPEN_HAND, t_norm)
            self._write_hand(q)
            if t_norm >= 1.0:
                print("[stage] done")
                self.done = True


def main():
    p = argparse.ArgumentParser(description="G1 Dex3 hand open/close test")
    p.add_argument("--iface", default="enx00e04c7015d3")
    p.add_argument("--open-only", action="store_true")
    p.add_argument("--open-duration", type=float, default=2.0)
    p.add_argument("--open-hold", type=float, default=2.0)
    p.add_argument("--close-duration", type=float, default=4.0,
                   help="Time to ramp from OPEN to CLOSED (slower = more pressure sensitivity)")
    p.add_argument("--close-hold", type=float, default=5.0)
    p.add_argument("--release-duration", type=float, default=2.0)
    p.add_argument("--hand-kp", type=float, default=5.0,
                   help="Hand motor KP (start low, Dex3 motors are small)")
    p.add_argument("--hand-kd", type=float, default=0.5)
    p.add_argument("--pressure-threshold", type=float, default=3000.0,
                   help="Pressure delta above baseline to trigger grip freeze")
    args = p.parse_args()

    print("=" * 60)
    print(" G1 DEX3 HAND TEST — OPEN / PRESSURE-GATED CLOSE")
    print("=" * 60)
    print(" This tests hand control independently from arm motion.")
    print(" Robot should be in Regular Mode (R1+X).")
    print(" Place an object in the hand before closing, or test empty.")
    print("=" * 60)

    ctrl = HandController(args)
    ctrl.init_channels()
    if ctrl.start():
        ctrl.run()


if __name__ == "__main__":
    main()