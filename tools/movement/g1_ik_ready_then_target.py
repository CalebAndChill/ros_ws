#!/usr/bin/env python3
"""
G1 IK Ready-Then-Target with Dex3 hand grasp.

Full pipeline:
  1. Solve Pinocchio IK for right_wrist_yaw_link
  2. Connect SDK2 (arm + hand)
  3. Arm: current → clearance roll → READY pose
  4. Hand: open fingers during hold_ready
  5. Arm: READY → IK target
  6. Hand: close fingers (pressure-gated) during hold_target
  7. Release arm_sdk (hand stays gripped)

Modes:
  --dry-run    : IK only, no robot
  --ready-only : arm to READY + hand open, then release
  --no-hand    : arm only, skip hand control

Location: ~/ros2_ws/tools/movement/g1_ik_ready_then_target.py
"""

import os, sys, site

HOME = os.path.expanduser("~")
UNITREE_REPO = os.path.join(HOME, "unitree_sdk2_python")
VENV_SITE = os.path.join(HOME, "unitree_sdk2_venv", "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
if os.path.isdir(UNITREE_REPO) and UNITREE_REPO not in sys.path:
    sys.path.insert(0, UNITREE_REPO)
if os.path.isdir(VENV_SITE):
    site.addsitedir(VENV_SITE)

import argparse, threading, time
from typing import Dict, List, Optional
import numpy as np

# ═══ Pinocchio IK (unchanged) ═══════════════════════════════════════

URDF = "/home/caleb/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf"
MESH_DIRS = ["/home/caleb/gazebo_g1_ws/src"]
TORSO_FRAME = "torso_link"
EE_FRAME = "right_wrist_yaw_link"

CONTROL_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

READY_REDUCED = {
    "right_shoulder_pitch_joint": 0.1, "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.2,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

PIN_TO_SDK_NAME = {
    "right_shoulder_pitch_joint": "right_shoulder_pitch",
    "right_shoulder_roll_joint": "right_shoulder_roll",
    "right_shoulder_yaw_joint": "right_shoulder_yaw",
    "right_elbow_joint": "right_elbow",
    "right_wrist_roll_joint": "right_wrist_roll",
    "right_wrist_pitch_joint": "right_wrist_pitch",
    "right_wrist_yaw_joint": "right_wrist_yaw",
}


def build_reduced_model():
    import pinocchio as pin
    model, _, _ = pin.buildModelsFromUrdf(URDF, MESH_DIRS)
    q0 = pin.neutral(model)
    keep = set(CONTROL_JOINTS)
    lock_ids = [model.getJointId(j) for j in model.names
                if j != "universe" and j not in keep and model.getJointId(j) != 0]
    return pin.buildReducedModel(model, lock_ids, q0)


def build_ready_q_reduced(model):
    import pinocchio as pin
    q = pin.neutral(model)
    for i, name in enumerate(model.names[1:], start=0):
        if name in READY_REDUCED:
            q[i] = READY_REDUCED[name]
    return q


def fk(model, data, q, frame_name):
    import pinocchio as pin
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame_name)]


def clamp_to_limits(model, q):
    q_c = q.copy()
    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    for i in range(len(q_c)):
        if lo[i] > -1e10 and hi[i] < 1e10:
            q_c[i] = np.clip(q_c[i], lo[i], hi[i])
    return q_c


def solve_position_ik(model, data, frame_name, q_init, q_nominal, target_pos_world,
                      max_iters=300, tol=2e-3, damping=1e-3, posture_gain=0.02, max_step=0.10):
    import pinocchio as pin
    fid = model.getFrameId(frame_name)
    q = q_init.copy()
    for i in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err = target_pos_world - data.oMf[fid].translation
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return True, q, i, err_norm
        J = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        J_pinv = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), np.eye(3))
        dq = J_pinv @ err + (np.eye(model.nv) - J_pinv @ J) @ (posture_gain * (q_nominal - q))
        n = np.linalg.norm(dq)
        if n > max_step: dq *= max_step / n
        q = clamp_to_limits(model, pin.integrate(model, q, dq))
    return False, q, max_iters, err_norm


def solve_ik_for_target(target_forward, target_left, target_up):
    model = build_reduced_model()
    data = model.createData()
    q_ready = build_ready_q_reduced(model)
    torso_pose = fk(model, data, q_ready, TORSO_FRAME)
    ee_in_torso = torso_pose.actInv(fk(model, data, q_ready, EE_FRAME))
    target_torso = np.array([target_forward, target_left, target_up], dtype=float)
    target_world = torso_pose.rotation @ target_torso + torso_pose.translation
    print(f"  READY wrist in torso frame: {ee_in_torso.translation}")
    print(f"  Target in torso frame:      [{target_forward:.4f}, {target_left:.4f}, {target_up:.4f}]")
    ok, q_sol, iters, pos_err = solve_position_ik(
        model, data, EE_FRAME, q_ready, q_ready, target_world)
    if not ok:
        print(f"  IK FAILED after {iters} iters, error = {pos_err:.6f} m")
        return None
    final_torso = torso_pose.actInv(fk(model, data, q_sol, EE_FRAME))
    print(f"  IK success in {iters} iters, error = {pos_err:.6f} m")
    print(f"  Solved wrist in torso frame: {final_torso.translation}")
    result = {}
    for i, pin_name in enumerate(model.names[1:], start=0):
        if pin_name in PIN_TO_SDK_NAME:
            result[PIN_TO_SDK_NAME[pin_name]] = float(q_sol[i])
    return result


# ═══ Hand poses (measured from real Dex3-1) ══════════════════════════

HAND_NAMES = ["thumb_abd", "thumb_f0", "thumb_f1", "fL0", "fL1", "fR0", "fR1"]
OPEN_HAND  = [+0.012, +0.577, -0.029, -0.085, -0.069, -0.017, -0.082]
CLOSED_HAND = [+0.012, -1.009, -1.650, +1.488, +1.671, +1.493, +1.686]
NUM_HAND = 7
NUM_SENSORS = 9
INACTIVE_PRESS = 50000.0

# ═══ SDK2 Controller ═════════════════════════════════════════════════

RIGHT_ARM_NAMES = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
LEFT_ARM_INDICES = [15, 16, 17, 18, 19, 20, 21]
RIGHT_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]
WEIGHT_SLOT = 29


class Controller:
    def __init__(self, iface, ready_q, ik_target_q, per_joint_kp, kp_hold, kd,
                 weight_ramp_sec, clearance_roll, clearance_duration,
                 ready_duration, ready_hold, target_duration, target_hold,
                 release_sec, ik_gradual_step, ready_only, log_errors,
                 use_hand, hand_kp, hand_kd, hand_close_duration, pressure_threshold):

        from unitree_sdk2py.core.channel import (
            ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandState_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.utils.thread import RecurrentThread

        # Try to import HandCmd_ default constructor
        try:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
            self._make_hand_cmd = lambda: unitree_hg_msg_dds__HandCmd_()
        except ImportError:
            # Fallback: construct manually
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, MotorCmd_
            def _make():
                motors = [MotorCmd_() for _ in range(NUM_HAND)]
                return HandCmd_(motor_cmd=motors, reserve=0)
            self._make_hand_cmd = _make

        self.ready_q = ready_q
        self.ik_target_q = ik_target_q
        self.per_joint_kp = per_joint_kp
        self.kp_hold = kp_hold
        self.kd = kd
        self.weight_ramp_sec = weight_ramp_sec
        self.clearance_roll = clearance_roll
        self.clearance_duration = clearance_duration
        self.ready_duration = ready_duration
        self.ready_hold = ready_hold
        self.target_duration = target_duration
        self.target_hold = target_hold
        self.release_sec = release_sec
        self.ik_gradual_step = max(0.0, min(1.0, ik_gradual_step))
        self.ready_only = ready_only
        self.log_errors = log_errors
        self.use_hand = use_hand
        self.hand_kp = hand_kp
        self.hand_kd = hand_kd
        self.hand_close_duration = hand_close_duration
        self.pressure_threshold = pressure_threshold

        self.RecurrentThread = RecurrentThread
        self.control_dt = 0.02
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_state = False
        self.lock = threading.Lock()

        self.left_start_q = [0.0] * 7
        self.right_start_q = [0.0] * 7
        self.clearance_q = [0.0] * 7

        self.ik_cmd_q = None
        if ik_target_q is not None:
            self.ik_cmd_q = [ready_q[i] + ik_gradual_step * (ik_target_q[i] - ready_q[i])
                             for i in range(7)]

        # Hand state
        self.hand_state = None
        self.first_hand = False
        self.hand_start_q = [0.0] * NUM_HAND
        self.hand_current_cmd = list(CLOSED_HAND)  # start neutral
        self.pressure_baseline = None
        self.hand_frozen = False
        self.hand_frozen_q = None
        self.hand_open_done = False
        self.hand_close_started = False
        self.hand_close_t0 = 0.0

        self.stage = "wait"
        self.t_stage = 0.0
        self.done = False
        self.log_printed = False

        # Init channels
        ChannelFactoryInitialize(0, iface)
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

        if self.use_hand:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
            self.hand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
            self.hand_pub.Init()
            self.hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
            self.hand_sub.Init(self._on_hand_state, 10)

        self.thread = None

    def _on_state(self, msg):
        with self.lock:
            self.low_state = msg
            if not self.first_state:
                self.first_state = True
                self.left_start_q = [msg.motor_state[i].q for i in LEFT_ARM_INDICES]
                self.right_start_q = [msg.motor_state[i].q for i in RIGHT_ARM_INDICES]
                self.clearance_q = list(self.right_start_q)
                self.clearance_q[1] = self.clearance_roll

    def _on_hand_state(self, msg):
        with self.lock:
            self.hand_state = msg
            if not self.first_hand:
                self.first_hand = True
                self.hand_start_q = [msg.motor_state[i].q for i in range(NUM_HAND)]

    # ── Arm helpers ───────────────────────────────────────────────────

    def _set_motor(self, idx, q, kp, kd):
        m = self.low_cmd.motor_cmd[idx]
        m.mode = 1; m.q = q; m.dq = 0.0; m.tau = 0.0; m.kp = kp; m.kd = kd

    def _write_arms(self, right_q_7):
        for i, idx in enumerate(LEFT_ARM_INDICES):
            self._set_motor(idx, self.left_start_q[i], self.kp_hold, self.kd)
        for i, idx in enumerate(RIGHT_ARM_INDICES):
            self._set_motor(idx, right_q_7[i], self.per_joint_kp[i], self.kd)

    def _blend(self, s, e, t):
        t = max(0.0, min(1.0, t))
        return [s[i] + t * (e[i] - s[i]) for i in range(len(s))]

    def _publish_arm(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _log_joint_errors(self, target_q, stage_name):
        with self.lock:
            if self.low_state is None: return
            msg = self.low_state
        print(f"\n{'=' * 78}\n Joint errors during {stage_name}\n{'=' * 78}")
        print(f"{'idx':>4}  {'name':<22}  {'target':>10}  {'actual':>10}  {'error':>10}  {'dq':>10}  {'kp':>6}")
        max_err = 0.0
        for i, idx in enumerate(RIGHT_ARM_INDICES):
            a = msg.motor_state[idx].q; dq = msg.motor_state[idx].dq
            e = a - target_q[i]; max_err = max(max_err, abs(e))
            print(f"{idx:>4}  {RIGHT_ARM_NAMES[i]:<22}  {target_q[i]:+10.4f}  {a:+10.4f}  {e:+10.4f}  {dq:+10.4f}  {self.per_joint_kp[i]:6.1f}")
        print(f" Max |error| = {max_err:.4f} rad ({max_err * 57.3:.1f} deg)\n{'=' * 78}\n")

    # ── Hand helpers ──────────────────────────────────────────────────

    def _write_hand(self, q_7):
        if not self.use_hand: return
        cmd = self._make_hand_cmd()
        for i in range(NUM_HAND):
            cmd.motor_cmd[i].q = q_7[i]
            cmd.motor_cmd[i].kp = self.hand_kp
            cmd.motor_cmd[i].kd = self.hand_kd
            cmd.motor_cmd[i].mode = 1
        self.hand_pub.Write(cmd)

    def _get_active_pressures(self):
        with self.lock:
            msg = self.hand_state
        if msg is None: return []
        readings = []
        for attr in ("press_sensor_state", "pressure_state", "press_state"):
            if hasattr(msg, attr):
                press = getattr(msg, attr)
                try:
                    for i in range(min(NUM_SENSORS, len(press))):
                        ps = press[i]
                        for a in ("pressure", "value", "data"):
                            if hasattr(ps, a):
                                vals = getattr(ps, a)
                                if hasattr(vals, '__iter__'):
                                    readings.extend(v for v in vals if v > INACTIVE_PRESS)
                                elif vals > INACTIVE_PRESS:
                                    readings.append(vals)
                                break
                except: pass
                break
        return readings

    def _capture_pressure_baseline(self):
        self.pressure_baseline = self._get_active_pressures()
        if self.pressure_baseline:
            print(f"  Pressure baseline: {len(self.pressure_baseline)} channels, "
                  f"range {min(self.pressure_baseline):.0f}-{max(self.pressure_baseline):.0f}")
        else:
            print("  WARNING: No pressure readings. Closing will be unguarded.")

    def _pressure_exceeded(self):
        if self.pressure_baseline is None: return False
        current = self._get_active_pressures()
        if len(current) != len(self.pressure_baseline): return False
        for i in range(len(current)):
            if current[i] - self.pressure_baseline[i] > self.pressure_threshold:
                return True
        return False

    # ── Hand actions (called from _tick) ──────────────────────────────

    def _hand_tick_open(self, elapsed_in_stage):
        """Ramp hand to OPEN over 2s, called during hold_ready."""
        if not self.use_hand or self.hand_open_done: return
        open_dur = 2.0
        t = min(1.0, elapsed_in_stage / open_dur)
        q = self._blend(self.hand_start_q, OPEN_HAND, t)
        self._write_hand(q)
        if t >= 1.0 and not self.hand_open_done:
            self.hand_open_done = True
            self._capture_pressure_baseline()
            print("[hand] open complete, baseline captured")

    def _hand_tick_close(self, now):
        """Gradually close hand, freeze on pressure. Called during hold_target."""
        if not self.use_hand: return
        if not self.hand_close_started:
            self.hand_close_started = True
            self.hand_close_t0 = now
            print("[hand] starting pressure-gated close")

        if self.hand_frozen:
            self._write_hand(self.hand_frozen_q)
            return

        t = min(1.0, (now - self.hand_close_t0) / self.hand_close_duration)
        q = self._blend(OPEN_HAND, CLOSED_HAND, t)

        if self._pressure_exceeded():
            self.hand_frozen = True
            self.hand_frozen_q = list(q)
            print(f"[hand] PRESSURE contact at {t*100:.0f}% close! Fingers frozen.")
            for i, n in enumerate(HAND_NAMES):
                print(f"  {n:<10} frozen at {q[i]:+.4f}")
        else:
            self._write_hand(q)
            if t >= 1.0:
                self.hand_frozen = True
                self.hand_frozen_q = list(CLOSED_HAND)
                print("[hand] fully closed (no pressure trigger)")

    def _hand_tick_hold(self):
        """Keep hand at last commanded position."""
        if not self.use_hand: return
        if self.hand_frozen and self.hand_frozen_q:
            self._write_hand(self.hand_frozen_q)
        elif self.hand_open_done:
            self._write_hand(OPEN_HAND)

    # ── Main loop ─────────────────────────────────────────────────────

    def start(self):
        print("Waiting for rt/lowstate ...")
        while not self.first_state:
            time.sleep(0.05)
        print("Got lowstate. Right arm current q:")
        for i, n in enumerate(RIGHT_ARM_NAMES):
            print(f"  [{RIGHT_ARM_INDICES[i]}] {n:<22} = {self.right_start_q[i]:+.4f}")

        if self.use_hand:
            print("Waiting for rt/dex3/right/state ...")
            t0 = time.time()
            while not self.first_hand:
                if time.time() - t0 > 5:
                    print("  WARNING: No hand state. Continuing without hand.")
                    self.use_hand = False
                    break
                time.sleep(0.05)
            if self.use_hand:
                print("Hand state received.")

        input("\nPress Enter to begin (Ctrl-C aborts)... ")
        self.stage = "weight_up"
        self.t_stage = time.time()
        self.thread = self.RecurrentThread(interval=self.control_dt,
            target=self._tick, name="g1_ik_ready_then_target")
        self.thread.Start()

    def _tick(self):
        if self.done: return
        now = time.time()

        if self.stage == "weight_up":
            r = max(0.0, min(1.0, (now - self.t_stage) / self.weight_ramp_sec))
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = r
            self._write_arms(self.right_start_q)
            self._publish_arm()
            if r >= 1.0:
                self.stage = "move_clearance"; self.t_stage = now
                print("[stage] weight=1.0, clearance roll")

        elif self.stage == "move_clearance":
            t = max(0.0, min(1.0, (now - self.t_stage) / self.clearance_duration))
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0
            self._write_arms(self._blend(self.right_start_q, self.clearance_q, t))
            self._publish_arm()
            if t >= 1.0:
                self.stage = "move_ready"; self.t_stage = now
                print("[stage] clearance done, moving to READY")

        elif self.stage == "move_ready":
            t = max(0.0, min(1.0, (now - self.t_stage) / self.ready_duration))
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0
            self._write_arms(self._blend(self.clearance_q, self.ready_q, t))
            self._publish_arm()
            if t >= 1.0:
                self.stage = "hold_ready"; self.t_stage = now; self.log_printed = False
                print("[stage] READY reached, holding + opening hand")

        elif self.stage == "hold_ready":
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0
            self._write_arms(self.ready_q)
            self._publish_arm()
            self._hand_tick_open(now - self.t_stage)
            if self.log_errors and not self.log_printed:
                if (now - self.t_stage) >= min(1.0, self.ready_hold * 0.5):
                    self._log_joint_errors(self.ready_q, "READY hold")
                    self.log_printed = True
            if (now - self.t_stage) >= self.ready_hold:
                if self.ready_only or self.ik_cmd_q is None:
                    self.stage = "release"; self.t_stage = now
                    print("[stage] ready-only, releasing")
                else:
                    self.stage = "move_target"; self.t_stage = now; self.log_printed = False
                    print("[stage] moving READY → IK target")

        elif self.stage == "move_target":
            t = max(0.0, min(1.0, (now - self.t_stage) / self.target_duration))
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0
            self._write_arms(self._blend(self.ready_q, self.ik_cmd_q, t))
            self._publish_arm()
            self._hand_tick_hold()  # keep hand open while arm moves
            if t >= 1.0:
                self.stage = "hold_target"; self.t_stage = now; self.log_printed = False
                print("[stage] IK target reached, closing hand")

        elif self.stage == "hold_target":
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0
            self._write_arms(self.ik_cmd_q)
            self._publish_arm()
            self._hand_tick_close(now)
            if self.log_errors and not self.log_printed:
                if (now - self.t_stage) >= min(2.0, self.target_hold * 0.5):
                    self._log_joint_errors(self.ik_cmd_q, "IK target hold")
                    self.log_printed = True
            if (now - self.t_stage) >= self.target_hold:
                self.stage = "release"; self.t_stage = now
                print("[stage] hold done, releasing arm (hand stays gripped)")

        elif self.stage == "release":
            r = max(0.0, min(1.0, (now - self.t_stage) / self.release_sec))
            self.low_cmd.motor_cmd[WEIGHT_SLOT].q = 1.0 - r
            last_q = self.ik_cmd_q if (self.ik_cmd_q and not self.ready_only) else self.ready_q
            self._write_arms(last_q)
            self._publish_arm()
            self._hand_tick_hold()  # keep hand gripped
            if r >= 1.0:
                print("[stage] arm_sdk released. Done.")
                self.done = True


# ═══ Args + Main ═════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="G1 IK + hand grasp")

    p.add_argument("--target-forward", type=float, default=0.12)
    p.add_argument("--target-left", type=float, default=-0.18)
    p.add_argument("--target-up", type=float, default=0.00)

    p.add_argument("--ready-shoulder-pitch", type=float, default=-0.20)
    p.add_argument("--ready-shoulder-roll", type=float, default=-0.07)
    p.add_argument("--ready-shoulder-yaw", type=float, default=-0.10)
    p.add_argument("--ready-elbow", type=float, default=-0.10)
    p.add_argument("--ready-wrist-roll", type=float, default=-0.04)
    p.add_argument("--ready-wrist-pitch", type=float, default=0.02)
    p.add_argument("--ready-wrist-yaw", type=float, default=-0.07)

    p.add_argument("--safe-shoulder-roll", type=float, default=-0.30)
    p.add_argument("--force-safe-shoulder-roll", action="store_true")
    p.add_argument("--ik-gradual-step", type=float, default=1.0)

    p.add_argument("--ws-min-forward", type=float, default=0.05)
    p.add_argument("--ws-max-forward", type=float, default=0.40)
    p.add_argument("--ws-min-left", type=float, default=-0.40)
    p.add_argument("--ws-max-left", type=float, default=-0.05)
    p.add_argument("--ws-min-up", type=float, default=-0.20)
    p.add_argument("--ws-max-up", type=float, default=0.25)

    p.add_argument("--kp-test", type=float, default=35.0)
    p.add_argument("--right-shoulder-roll-kp", type=float, default=45.0)
    p.add_argument("--right-elbow-kp", type=float, default=65.0)
    p.add_argument("--kp-hold", type=float, default=40.0)
    p.add_argument("--kd", type=float, default=1.5)

    p.add_argument("--weight-ramp", type=float, default=1.5)
    p.add_argument("--clearance-duration", type=float, default=3.0)
    p.add_argument("--ready-duration", type=float, default=8.0)
    p.add_argument("--ready-hold", type=float, default=3.0)
    p.add_argument("--target-duration", type=float, default=10.0)
    p.add_argument("--target-hold", type=float, default=8.0)
    p.add_argument("--release", type=float, default=2.0)

    # Hand
    p.add_argument("--no-hand", action="store_true", help="Skip hand control")
    p.add_argument("--hand-kp", type=float, default=10.0)
    p.add_argument("--hand-kd", type=float, default=0.5)
    p.add_argument("--hand-close-duration", type=float, default=4.0)
    p.add_argument("--pressure-threshold", type=float, default=3000.0)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ready-only", action="store_true")
    p.add_argument("--log-errors", action="store_true")
    p.add_argument("--iface", default="enx00e04c7015d3")

    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print(" G1 IK READY-THEN-TARGET + DEX3 HAND GRASP")
    print("=" * 78)

    # Workspace check
    fwd, lft, up = args.target_forward, args.target_left, args.target_up
    rejected = []
    if fwd < args.ws_min_forward: rejected.append(f"forward {fwd:+.3f} < {args.ws_min_forward:+.3f}")
    if fwd > args.ws_max_forward: rejected.append(f"forward {fwd:+.3f} > {args.ws_max_forward:+.3f}")
    if lft > args.ws_max_left: rejected.append(f"left {lft:+.3f} > {args.ws_max_left:+.3f} (wrong side)")
    if lft < args.ws_min_left: rejected.append(f"left {lft:+.3f} < {args.ws_min_left:+.3f}")
    if up < args.ws_min_up: rejected.append(f"up {up:+.3f} < {args.ws_min_up:+.3f} (below table)")
    if up > args.ws_max_up: rejected.append(f"up {up:+.3f} > {args.ws_max_up:+.3f}")
    if rejected:
        print("\n  [WORKSPACE REJECT]")
        for r in rejected: print(f"    - {r}")
        sys.exit(1)
    print("  Workspace check: PASSED")

    # Solve IK
    print("\n  Solving Pinocchio IK...")
    ik_result = solve_ik_for_target(fwd, lft, up)
    if ik_result is None:
        print("\n  IK failed."); sys.exit(1)

    print("\n  IK solution:")
    for n in RIGHT_ARM_NAMES: print(f"    {n:<22} = {ik_result[n]:+.5f}")

    if args.force_safe_shoulder_roll:
        raw = ik_result["right_shoulder_roll"]
        if raw > args.safe_shoulder_roll:
            print(f"\n  [SAFETY] shoulder_roll {raw:+.5f} → clamped to {args.safe_shoulder_roll:+.5f}")
            ik_result["right_shoulder_roll"] = args.safe_shoulder_roll

    ready_q = [args.ready_shoulder_pitch, args.ready_shoulder_roll, args.ready_shoulder_yaw,
               args.ready_elbow, args.ready_wrist_roll, args.ready_wrist_pitch, args.ready_wrist_yaw]
    ik_q = [ik_result[n] for n in RIGHT_ARM_NAMES]
    gs = max(0.0, min(1.0, args.ik_gradual_step))
    ik_cmd_q = [ready_q[i] + gs * (ik_q[i] - ready_q[i]) for i in range(7)]

    per_joint_kp = [args.kp_test] * 7
    per_joint_kp[1] = args.right_shoulder_roll_kp
    per_joint_kp[3] = args.right_elbow_kp

    use_hand = not args.no_hand

    # Print plan
    rest = [+0.288, -0.136, +0.004, +0.970, -0.123, +0.040, -0.009]
    print(f"\n{'=' * 78}\n PLAN\n{'=' * 78}")
    mode = 'DRY RUN' if args.dry_run else ('READY ONLY' if args.ready_only else 'READY → IK → GRASP')
    print(f"  Mode: {mode}   Hand: {'ON' if use_hand else 'OFF'}   IK step: {gs:.0%}")
    print(f"  Stages: weight({args.weight_ramp}s) → clearance({args.clearance_duration}s) → "
          f"ready({args.ready_duration}s) → hold({args.ready_hold}s)")
    if not args.ready_only:
        print(f"          → target({args.target_duration}s) → hold+close({args.target_hold}s) → release({args.release}s)")
    if use_hand:
        print(f"  Hand: kp={args.hand_kp} close={args.hand_close_duration}s pressure_thresh={args.pressure_threshold}")

    print(f"\n  {'joint':<22} {'~rest':>8} {'ready':>8} {'IKcmd':>8} {'Δr→rdy':>8} {'Δrdy→ik':>8} {'kp':>6}")
    for i, n in enumerate(RIGHT_ARM_NAMES):
        d1, d2 = ready_q[i]-rest[i], ik_cmd_q[i]-ready_q[i]
        print(f"  {n:<22} {rest[i]:+8.3f} {ready_q[i]:+8.3f} {ik_cmd_q[i]:+8.3f} {d1:+8.3f} {d2:+8.3f} {per_joint_kp[i]:6.1f}")
    print(f"{'=' * 78}")

    if args.dry_run:
        print("\n  DRY RUN complete."); sys.exit(0)

    # Execute
    print(f"\n  Connecting to robot...")
    ctrl = Controller(
        iface=args.iface, ready_q=ready_q, ik_target_q=ik_q if not args.ready_only else None,
        per_joint_kp=per_joint_kp, kp_hold=args.kp_hold, kd=args.kd,
        weight_ramp_sec=args.weight_ramp, clearance_roll=args.safe_shoulder_roll,
        clearance_duration=args.clearance_duration, ready_duration=args.ready_duration,
        ready_hold=args.ready_hold, target_duration=args.target_duration,
        target_hold=args.target_hold, release_sec=args.release,
        ik_gradual_step=args.ik_gradual_step, ready_only=args.ready_only,
        log_errors=args.log_errors, use_hand=use_hand,
        hand_kp=args.hand_kp, hand_kd=args.hand_kd,
        hand_close_duration=args.hand_close_duration,
        pressure_threshold=args.pressure_threshold)

    ctrl.start()
    try:
        while not ctrl.done: time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Releasing arm_sdk...")
        try:
            for _ in range(50):
                ctrl.low_cmd.motor_cmd[WEIGHT_SLOT].q = 0.0
                ctrl.low_cmd.crc = ctrl.crc.Crc(ctrl.low_cmd)
                ctrl.pub.Write(ctrl.low_cmd)
                time.sleep(0.02)
            print("  Weight released. Press L2+B if needed.")
        except: print("  PRESS L2+B NOW.")


if __name__ == "__main__":
    main()