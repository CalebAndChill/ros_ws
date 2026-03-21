import math
import argparse
from typing import List, Dict, Any

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


def signed_dir(value: float, pos_word: str, neg_word: str) -> str:
    if math.isnan(value):
        return "unknown"
    if abs(value) < 0.5:
        return "center"
    return f"{abs(value):.1f}° {pos_word if value > 0 else neg_word}"


class Yolo3DInspector(Node):
    def __init__(self, topic: str, hz: float, min_conf: float, top_k: int, sort_by: str):
        super().__init__("yolo_3d_inspector_clean")

        self.min_conf = min_conf
        self.top_k = top_k
        self.sort_by = sort_by
        self.latest: List[Dict[str, Any]] = []
        self.last_frame_id = ""

        qos = QoSProfile(depth=10)
        self.sub = self.create_subscription(DetectionArray, topic, self.cb, qos)

        period = 1.0 / hz if hz > 0 else 1.0
        self.timer = self.create_timer(period, self.print_summary)

        self.get_logger().info(
            f"Subscribed to {topic} ({MSG_FLAVOR}) | print rate={hz:.2f} Hz | "
            f"min_conf={min_conf:.2f} | top_k={top_k} | sort_by={sort_by}"
        )

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

                rng = math.sqrt(x * x + y * y + z * z) if not any(math.isnan(v) for v in [x, y, z]) else float("nan")
                az = deg(math.atan2(x, z)) if not any(math.isnan(v) for v in [x, z]) else float("nan")
                el = deg(math.atan2(-y, z)) if not any(math.isnan(v) for v in [y, z]) else float("nan")

                detections.append(
                    {
                        "id": det_id,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "score": score,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "range": rng,
                        "az": az,
                        "el": el,
                        "frame_id": frame_id,
                    }
                )

        else:
            # Fallback for vision_msgs
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

                rng = math.sqrt(x * x + y * y + z * z) if not any(math.isnan(v) for v in [x, y, z]) else float("nan")
                az = deg(math.atan2(x, z)) if not any(math.isnan(v) for v in [x, z]) else float("nan")
                el = deg(math.atan2(-y, z)) if not any(math.isnan(v) for v in [y, z]) else float("nan")

                detections.append(
                    {
                        "id": "",
                        "class_id": -1,
                        "class_name": str(label),
                        "score": score,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dx": dx,
                        "dy": dy,
                        "dz": dz,
                        "range": rng,
                        "az": az,
                        "el": el,
                        "frame_id": getattr(getattr(msg, "header", None), "frame_id", ""),
                    }
                )

        self.latest = detections
        if detections:
            self.last_frame_id = detections[0]["frame_id"]

    def print_summary(self):
        if not self.latest:
            self.get_logger().info("No detections")
            return

        detections = list(self.latest)

        if self.sort_by == "range":
            detections.sort(key=lambda d: d["range"] if not math.isnan(d["range"]) else 9999.0)
        elif self.sort_by == "z":
            detections.sort(key=lambda d: d["z"] if not math.isnan(d["z"]) else 9999.0)
        else:  # conf
            detections.sort(key=lambda d: d["score"], reverse=True)

        detections = detections[: self.top_k]

        self.get_logger().info(
            f"--- {len(self.latest)} detection(s), showing {len(detections)} | frame='{self.last_frame_id}' ---"
        )

        for i, d in enumerate(detections, start=1):
            az_text = signed_dir(d["az"], "right", "left")
            el_text = signed_dir(d["el"], "up", "down")

            det_id_text = f" [id={d['id']}]" if str(d["id"]) else ""

            self.get_logger().info(
                f"{i}. {d['class_name']}{det_id_text}  conf={d['score']:.2f}"
            )
            self.get_logger().info(
                f"   forward={d['z']:.2f} m | range={d['range']:.2f} m | "
                f"bearing={az_text} | elevation={el_text}"
            )
            self.get_logger().info(
                f"   size={d['dx']:.2f} x {d['dy']:.2f} x {d['dz']:.2f} m"
            )


def main():
    parser = argparse.ArgumentParser(description="Clean YOLO 3D inspector")
    parser.add_argument("--topic", default="/yolo/detections_3d", help="3D detections topic")
    parser.add_argument("--hz", type=float, default=2.0, help="How often to print summaries")
    parser.add_argument("--min-conf", type=float, default=0.50, help="Ignore detections below this confidence")
    parser.add_argument("--top-k", type=int, default=3, help="How many detections to show")
    parser.add_argument(
        "--sort-by",
        choices=["conf", "range", "z"],
        default="range",
        help="Sort shown detections by confidence, total range, or forward z distance",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Yolo3DInspector(
        topic=args.topic,
        hz=args.hz,
        min_conf=args.min_conf,
        top_k=args.top_k,
        sort_by=args.sort_by,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()