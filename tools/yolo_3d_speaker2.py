#!/usr/bin/env python3
import os
import sys
import site
import math
import time
import argparse
from typing import List, Dict, Any

# -------------------------------------------------------------------
# Let normal ROS Python see your Unitree SDK repo + venv packages
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

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# Try yolo_msgs first, then vision_msgs
MSG_FLAVOR = None
try:
    from yolo_msgs.msg import DetectionArray  # type: ignore
    MSG_FLAVOR = "yolo"
except Exception:
    try:
        from vision_msgs.msg import Detection3DArray as DetectionArray  # type: ignore
        MSG_FLAVOR = "vision"
    except Exception:
        raise RuntimeError(
            "Neither yolo_msgs/DetectionArray nor vision_msgs/Detection3DArray is available."
        )


def deg(x: float) -> float:
    return math.degrees(x)


def clean_name(name: str) -> str:
    return str(name).replace("_", " ").strip()


def fmt_m(x: float) -> str:
    if math.isnan(x):
        return "unknown"
    return f"{x:.1f}"


class Yolo3DSpeaker(Node):
    def __init__(
        self,
        topic: str,
        speech_enabled: bool,
        iface: str,
        speaker_id: int,
        mode: str,
        announce_policy: str,
        announce_every: float,
        repeat_same_after: float,
        min_conf: float,
        top_k: int,
        sort_by: str,
        include_size: bool,
        side_center_deg: float,
        side_slight_deg: float,
        speech_char_rate: float,
        speech_base_sec: float,
    ):
        super().__init__("yolo_3d_speaker")

        self.speech_enabled = speech_enabled
        self.iface = iface
        self.speaker_id = speaker_id
        self.mode = mode
        self.announce_policy = announce_policy
        self.announce_every = announce_every
        self.repeat_same_after = repeat_same_after
        self.min_conf = min_conf
        self.top_k = top_k
        self.sort_by = sort_by
        self.include_size = include_size
        self.side_center_deg = side_center_deg
        self.side_slight_deg = side_slight_deg
        self.speech_char_rate = speech_char_rate
        self.speech_base_sec = speech_base_sec

        self.latest: List[Dict[str, Any]] = []
        self.last_frame_id = ""

        self.audio = None
        self.speech_ready = False

        self.pending_lines: List[str] = []
        self.last_build_time = 0.0
        self.last_batch_key = ""
        self.last_batch_time = 0.0
        self.speaking_until = 0.0

        if self.speech_enabled:
            self.init_audio()

        qos = QoSProfile(depth=10)
        self.sub = self.create_subscription(DetectionArray, topic, self.cb, qos)

        self.timer = self.create_timer(0.25, self.tick)

        self.get_logger().info(
            f"Subscribed to {topic} ({MSG_FLAVOR}) | "
            f"speech_enabled={self.speech_enabled} | mode={self.mode} | "
            f"announce_policy={self.announce_policy} | announce_every={self.announce_every:.1f}s | "
            f"min_conf={self.min_conf:.2f} | top_k={self.top_k} | sort_by={self.sort_by}"
        )

    # ---------------------------------------------------------------
    # Unitree audio init
    # ---------------------------------------------------------------
    def init_audio(self):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            ChannelFactoryInitialize(0, self.iface)

            self.audio = AudioClient()
            self.audio.SetTimeout(10.0)
            self.audio.Init()

            self.speech_ready = True
            self.get_logger().info(
                f"Unitree audio ready on iface={self.iface}, speaker_id={self.speaker_id}"
            )
        except Exception as e:
            self.speech_ready = False
            self.get_logger().error(f"Failed to initialize Unitree audio: {e}")
            self.get_logger().warn("Continuing in log-only mode.")
            self.speech_enabled = False

    # ---------------------------------------------------------------
    # Detection parsing
    # ---------------------------------------------------------------
    def cb(self, msg: DetectionArray):
        detections: List[Dict[str, Any]] = []

        if MSG_FLAVOR == "yolo":
            for det in getattr(msg, "detections", []):
                cls_id = getattr(det, "class_id", -1)
                cls_name = getattr(det, "class_name", "") or f"class_{cls_id}"
                score = float(getattr(det, "score", 0.0))
                det_id = getattr(det, "id", "")

                if score < self.min_conf:
                    continue

                frame_id = (
                    getattr(det, "frame_id", "")
                    or getattr(getattr(det, "bbox3d", None), "frame_id", "")
                    or getattr(getattr(msg, "header", None), "frame_id", "")
                )

                bbox3d = getattr(det, "bbox3d", None)
                x = y = z = dx = dy = dz = float("nan")
                if bbox3d:
                    center3d = getattr(bbox3d, "center", None)
                    size3d = getattr(bbox3d, "size", None)

                    if center3d:
                        pos3d = getattr(center3d, "position", None)
                        if pos3d:
                            x = float(getattr(pos3d, "x", float("nan")))
                            y = float(getattr(pos3d, "y", float("nan")))
                            z = float(getattr(pos3d, "z", float("nan")))

                    if size3d:
                        dx = float(getattr(size3d, "x", float("nan")))
                        dy = float(getattr(size3d, "y", float("nan")))
                        dz = float(getattr(size3d, "z", float("nan")))

                rng = (
                    math.sqrt(x * x + y * y + z * z)
                    if not any(math.isnan(v) for v in [x, y, z])
                    else float("nan")
                )
                az = deg(math.atan2(x, z)) if not any(math.isnan(v) for v in [x, z]) else float("nan")

                detections.append(
                    {
                        "id": det_id,
                        "class_id": cls_id,
                        "class_name": clean_name(cls_name),
                        "score": score,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "range": rng,
                        "az": az,
                        "frame_id": frame_id,
                    }
                )

        else:
            for det in getattr(msg, "detections", []):
                label = "unknown"
                score = 0.0
                results = getattr(det, "results", [])
                if results:
                    hyp = getattr(results[0], "hypothesis", results[0])
                    label = getattr(hyp, "class_id", getattr(hyp, "id", "unknown"))
                    score = float(getattr(hyp, "score", 0.0))

                if score < self.min_conf:
                    continue

                bbox3d = getattr(det, "bbox", None)
                x = y = z = dx = dy = dz = float("nan")
                if bbox3d:
                    center = getattr(bbox3d, "center", None)
                    size = getattr(bbox3d, "size", None)
                    if center and getattr(center, "position", None):
                        pos = center.position
                        x, y, z = float(pos.x), float(pos.y), float(pos.z)
                    if size:
                        dx, dy, dz = float(size.x), float(size.y), float(size.z)

                rng = (
                    math.sqrt(x * x + y * y + z * z)
                    if not any(math.isnan(v) for v in [x, y, z])
                    else float("nan")
                )
                az = deg(math.atan2(x, z)) if not any(math.isnan(v) for v in [x, z]) else float("nan")

                detections.append(
                    {
                        "id": "",
                        "class_id": -1,
                        "class_name": clean_name(str(label)),
                        "score": score,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "range": rng,
                        "az": az,
                        "frame_id": getattr(getattr(msg, "header", None), "frame_id", ""),
                    }
                )

        self.latest = detections
        if detections:
            self.last_frame_id = detections[0]["frame_id"]

    # ---------------------------------------------------------------
    # Description helpers
    # ---------------------------------------------------------------
    def side_text(self, az: float) -> str:
        if math.isnan(az):
            return "unknown"

        if abs(az) <= self.side_center_deg:
            return "center"
        if abs(az) <= self.side_slight_deg:
            return "slightly right" if az > 0 else "slightly left"
        return "right" if az > 0 else "left"

    def distance_text(self, d: Dict[str, Any]) -> str:
        if not math.isnan(d["z"]):
            return f"{fmt_m(d['z'])} meters ahead"
        if not math.isnan(d["range"]):
            return f"{fmt_m(d['range'])} meters away"
        return "distance unknown"

    def size_text(self, d: Dict[str, Any]) -> str:
        if any(math.isnan(v) for v in [d["dx"], d["dy"], d["dz"]]):
            return "size unknown"

        dx_cm = int(round(d["dx"] * 100))
        dy_cm = int(round(d["dy"] * 100))
        dz_cm = int(round(d["dz"] * 100))
        return f"size about {dx_cm} by {dy_cm} by {dz_cm} centimeters"

    def describe_detection(self, d: Dict[str, Any]) -> str:
        name = d["class_name"]
        side = self.side_text(d["az"])
        dist = self.distance_text(d)

        if self.mode == "simple":
            return name

        if self.mode == "directional":
            return f"{name}, {side}"

        if self.mode == "standard":
            if side == "center":
                return f"{name}, {dist}, center"
            return f"{name}, {dist}, {side}"

        # detailed
        parts = [name, dist]
        parts.append("center" if side == "center" else side)

        if self.include_size:
            parts.append(self.size_text(d))

        conf_pct = int(round(d["score"] * 100))
        parts.append(f"confidence {conf_pct} percent")
        return ", ".join(parts)

    def sort_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detections = list(detections)

        if self.sort_by == "range":
            detections.sort(key=lambda d: d["range"] if not math.isnan(d["range"]) else 9999.0)
        elif self.sort_by == "z":
            detections.sort(key=lambda d: d["z"] if not math.isnan(d["z"]) else 9999.0)
        else:  # conf
            detections.sort(key=lambda d: d["score"], reverse=True)

        return detections

    def build_lines(self) -> List[str]:
        if not self.latest:
            return []

        detections = self.sort_detections(self.latest)

        if self.announce_policy == "best":
            return [self.describe_detection(detections[0])]

        detections = detections[: self.top_k]
        return [self.describe_detection(d) for d in detections]

    # ---------------------------------------------------------------
    # Speech helpers
    # ---------------------------------------------------------------
    def estimate_tts_seconds(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0
        return max(2.0, self.speech_base_sec + len(text) / self.speech_char_rate)

    def say(self, text: str) -> bool:
        self.get_logger().info(f"Announcement: {text}")

        now = time.monotonic()

        if now < self.speaking_until:
            self.get_logger().info("Still speaking previous line, skipping overlap.")
            return False

        if not self.speech_enabled:
            self.speaking_until = now + self.estimate_tts_seconds(text)
            return True

        if not self.speech_ready or not self.audio:
            return False

        try:
            ret = self.audio.TtsMaker(text, self.speaker_id)
            self.get_logger().info(f"TTS return: {ret}")

            if ret == 0:
                self.speaking_until = time.monotonic() + self.estimate_tts_seconds(text)
                return True

            return False
        except Exception as e:
            self.get_logger().error(f"TTS failed: {e}")
            return False

    # ---------------------------------------------------------------
    # Main timer
    # ---------------------------------------------------------------
    def tick(self):
        now = time.monotonic()

        if now < self.speaking_until:
            return

        if self.pending_lines:
            line = self.pending_lines.pop(0)
            self.say(line)
            return

        if now - self.last_build_time < self.announce_every:
            return

        lines = self.build_lines()
        self.last_build_time = now

        if not lines:
            return

        batch_key = " || ".join(lines)
        if batch_key == self.last_batch_key and (now - self.last_batch_time) < self.repeat_same_after:
            return

        self.last_batch_key = batch_key
        self.last_batch_time = now

        self.pending_lines = list(lines)

        if self.pending_lines:
            line = self.pending_lines.pop(0)
            self.say(line)


def main():
    parser = argparse.ArgumentParser(description="YOLO 3D speech announcer for Unitree G1")

    parser.add_argument("--topic", default="/yolo/detections_3d", help="3D detections topic")

    parser.add_argument("--speech-enabled", action="store_true", help="Actually speak via G1 audio")
    parser.add_argument("--iface", default="enx00e04c7015d3", help="Robot ethernet interface for Unitree SDK")
    parser.add_argument("--speaker-id", type=int, default=1, help="Speaker/voice id for TTS (English usually 1)")

    parser.add_argument(
        "--mode",
        choices=["simple", "directional", "standard", "detailed"],
        default="standard",
        help="Amount/type of spoken detail",
    )
    parser.add_argument(
        "--announce-policy",
        choices=["best", "all"],
        default="best",
        help="Speak only one detection or all shown detections",
    )
    parser.add_argument("--announce-every", type=float, default=4.0, help="Minimum seconds between fresh detection batches")
    parser.add_argument(
        "--repeat-same-after",
        type=float,
        default=10.0,
        help="Minimum seconds before repeating the exact same batch",
    )

    parser.add_argument("--min-conf", type=float, default=0.50, help="Ignore detections below this confidence")
    parser.add_argument("--top-k", type=int, default=3, help="When using --announce-policy all, max detections to mention")
    parser.add_argument(
        "--sort-by",
        choices=["conf", "range", "z"],
        default="conf",
        help="How to choose/arrange detections",
    )

    parser.add_argument("--include-size", action="store_true", help="Include 3D size in detailed mode")
    parser.add_argument("--side-center-deg", type=float, default=8.0, help="Within this azimuth = center")
    parser.add_argument("--side-slight-deg", type=float, default=20.0, help="Within this azimuth = slightly left/right")

    parser.add_argument(
        "--speech-char-rate",
        type=float,
        default=10.0,
        help="Estimated characters per second for TTS timing",
    )
    parser.add_argument(
        "--speech-base-sec",
        type=float,
        default=1.5,
        help="Extra fixed seconds added to each spoken line",
    )

    args = parser.parse_args()

    rclpy.init()
    node = Yolo3DSpeaker(
        topic=args.topic,
        speech_enabled=args.speech_enabled,
        iface=args.iface,
        speaker_id=args.speaker_id,
        mode=args.mode,
        announce_policy=args.announce_policy,
        announce_every=args.announce_every,
        repeat_same_after=args.repeat_same_after,
        min_conf=args.min_conf,
        top_k=args.top_k,
        sort_by=args.sort_by,
        include_size=args.include_size,
        side_center_deg=args.side_center_deg,
        side_slight_deg=args.side_slight_deg,
        speech_char_rate=args.speech_char_rate,
        speech_base_sec=args.speech_base_sec,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()