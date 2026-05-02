#!/usr/bin/env python3
import os
import sys
import site
import time
import argparse
import threading
from typing import Dict, List

# -------------------------------------------------------------------
# Let normal Python see your Unitree SDK repo + venv packages
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

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5

    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

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

    kNotUsedJoint = 29  # arm_sdk weight / enable


# ARM ONLY: matches your latest Pinocchio reduced model / real-test intent.
CONTROL_JOINTS: List[int] = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

CONTROL_NAMES: List[str] = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

PRESETS: Dict[str, Dict[str, float]] = {
    # cautious first real test
    "ready_small": {
        "right_shoulder_pitch": 0.15,
        "right_shoulder_roll": -0.10,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.35,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
    },
    # closer to your simulation arm-only ready pose
    "ready_sim": {
        "right_shoulder_pitch": 0.10,
        "right_shoulder_roll": -0.20,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.20,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
    },
    # slightly more lifted / bent arm, still arm-only
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


class G1RightArmOnlySDK2Controller:
    def __init__(
        self,
        iface: str,
        preset: str,
        duration: float,
        hold_sec: float,
        release_sec: float,
        kp: float,
        kd: float,
        start_delay: float,
        no_release: bool,
        overrides: Dict[str, float],
    ):
        self.iface = iface
        self.preset = preset
        self.duration = max(0.5, duration)
        self.hold_sec = max(0.0, hold_sec)
        self.release_sec = max(0.5, release_sec)
        self.kp = kp
        self.kd = kd
        self.start_delay = max(0.0, start_delay)
        self.no_release = no_release
        self.overrides = overrides

        self.control_dt = 0.02
        self.crc = CRC()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_low_state = False
        self.state_lock = threading.Lock()

        self.start_q = [0.0] * len(CONTROL_JOINTS)
        self.target_q = [0.0] * len(CONTROL_JOINTS)

        self.stage = "wait_state"
        self.stage_start_time = 0.0
        self.done = False

        self.arm_pub = None
        self.lowstate_sub = None
        self.thread = None

        self._build_target_q()

    def _build_target_q(self):
        base = PRESETS[self.preset].copy()
        base.update(self.overrides)
        self.target_q = [base[name] for name in CONTROL_NAMES]

    def init_channels(self):
        ChannelFactoryInitialize(0, self.iface)

        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()

        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        with self.state_lock:
            self.low_state = msg
            if not self.first_low_state:
                self.first_low_state = True
                self.start_q = [msg.motor_state[j].q for j in CONTROL_JOINTS]
                print("LowState received.")
                print("Current right-arm start pose:")
                for name, q in zip(CONTROL_NAMES, self.start_q):
                    print(f"  {name:22s} {q: .4f}")
                print("\nTarget pose:")
                for name, q in zip(CONTROL_NAMES, self.target_q):
                    print(f"  {name:22s} {q: .4f}")

    def start(self):
        print("Waiting for rt/lowstate ...")
        while not self.first_low_state:
            time.sleep(0.1)

        if self.start_delay > 0.0:
            print(f"Waiting extra {self.start_delay:.1f}s before moving ...")
            time.sleep(self.start_delay)

        self.stage = "move"
        self.stage_start_time = time.time()

        self.thread = RecurrentThread(
            interval=self.control_dt,
            target=self.control_loop,
            name="g1_right_arm_only_sdk2_control",
        )
        self.thread.Start()

    @staticmethod
    def blend(a: float, b: float, ratio: float) -> float:
        r = max(0.0, min(1.0, ratio))
        return (1.0 - r) * a + r * b

    def set_joint_cmd(self, joint: int, q: float, kp: float, kd: float):
        self.low_cmd.motor_cmd[joint].q = q
        self.low_cmd.motor_cmd[joint].dq = 0.0
        self.low_cmd.motor_cmd[joint].tau = 0.0
        self.low_cmd.motor_cmd[joint].kp = kp
        self.low_cmd.motor_cmd[joint].kd = kd

    def control_loop(self):
        if self.done:
            return

        now = time.time()

        with self.state_lock:
            low_state = self.low_state

        if low_state is None:
            return

        if self.stage == "move":
            ratio = max(0.0, min(1.0, (now - self.stage_start_time) / self.duration))
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

            for i, joint in enumerate(CONTROL_JOINTS):
                q = self.blend(self.start_q[i], self.target_q[i], ratio)
                self.set_joint_cmd(joint, q, self.kp, self.kd)

            if ratio >= 1.0:
                if self.hold_sec > 0.0:
                    self.stage = "hold"
                    self.stage_start_time = now
                elif self.no_release:
                    self.stage = "hold_forever"
                else:
                    self.stage = "release"
                    self.stage_start_time = now

        elif self.stage == "hold":
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
            for i, joint in enumerate(CONTROL_JOINTS):
                self.set_joint_cmd(joint, self.target_q[i], self.kp, self.kd)

            if (now - self.stage_start_time) >= self.hold_sec:
                if self.no_release:
                    self.stage = "hold_forever"
                else:
                    self.stage = "release"
                    self.stage_start_time = now

        elif self.stage == "hold_forever":
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0
            for i, joint in enumerate(CONTROL_JOINTS):
                self.set_joint_cmd(joint, self.target_q[i], self.kp, self.kd)

        elif self.stage == "release":
            ratio = max(0.0, min(1.0, (now - self.stage_start_time) / self.release_sec))
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - ratio
            for i, joint in enumerate(CONTROL_JOINTS):
                self.set_joint_cmd(joint, self.target_q[i], self.kp, self.kd)
            if ratio >= 1.0:
                self.done = True
                self.stage = "done"
                print("Done. arm_sdk released.")

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_pub.Write(self.low_cmd)


def parse_args():
    parser = argparse.ArgumentParser(
        description="G1 right-arm-only SDK2 position controller (real robot replacement for sim arm-only control)"
    )
    parser.add_argument("--iface", default="enx00e04c7015d3", help="Robot ethernet interface")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="ready_small")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--hold-sec", type=float, default=2.0)
    parser.add_argument("--release-sec", type=float, default=1.5)
    parser.add_argument("--kp", type=float, default=45.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--no-release", action="store_true")

    parser.add_argument("--right-shoulder-pitch", type=float, default=None)
    parser.add_argument("--right-shoulder-roll", type=float, default=None)
    parser.add_argument("--right-shoulder-yaw", type=float, default=None)
    parser.add_argument("--right-elbow", type=float, default=None)
    parser.add_argument("--right-wrist-roll", type=float, default=None)
    parser.add_argument("--right-wrist-pitch", type=float, default=None)
    parser.add_argument("--right-wrist-yaw", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
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

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("Recommended first test: --preset ready_small")
    print("This version is ARM ONLY: no waist_yaw command is sent.")
    print("If low-level control conflicts with higher-level motion, stop the higher-level motion service first.")
    input("Press Enter to continue... ")

    ctrl = G1RightArmOnlySDK2Controller(
        iface=args.iface,
        preset=args.preset,
        duration=args.duration,
        hold_sec=args.hold_sec,
        release_sec=args.release_sec,
        kp=args.kp,
        kd=args.kd,
        start_delay=args.start_delay,
        no_release=args.no_release,
        overrides=overrides,
    )
    ctrl.init_channels()
    ctrl.start()

    try:
        while True:
            time.sleep(0.5)
            if ctrl.done:
                break
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
