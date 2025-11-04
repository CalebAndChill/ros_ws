#!/usr/bin/env python3
# Prints per-object: class/id/score, 2D bbox (px), 3D center (m), 3D size (m),
# forward distance z, Euclidean range, azimuth/elevation (deg), mask size, frame_id.
# Optical-frame angles: az = atan2(x, z), el = atan2(-y, z)  (x right, y DOWN, z forward).

import math
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# Try yolo_msgs first, then vision_msgs for compatibility
MSG_FLAVOR = None
try:
    from yolo_msgs.msg import DetectionArray  # type: ignore
    MSG_FLAVOR = "yolo"
except Exception:
    try:
        from vision_msgs.msg import Detection3DArray as DetectionArray  # type: ignore
        MSG_FLAVOR = "vision"
    except Exception as e:
        raise RuntimeError(
            "Neither yolo_msgs/DetectionArray nor vision_msgs/Detection3DArray is available."
        )

def deg(x):
    return math.degrees(x)

class Yolo3DInspector(Node):
    def __init__(self, topic):
        super().__init__('yolo_3d_inspector')
        qos = QoSProfile(depth=10)
        self.sub = self.create_subscription(DetectionArray, topic, self.cb, qos)
        self.get_logger().info(f"Subscribed to {topic} ({MSG_FLAVOR})")

    def cb(self, msg: DetectionArray):
        # yolo_msgs layout
        if MSG_FLAVOR == "yolo":
            for i, det in enumerate(getattr(msg, 'detections', [])):
                cls_id    = getattr(det, 'class_id', -1)
                cls_name  = getattr(det, 'class_name', '')
                score     = float(getattr(det, 'score', 0.0))
                det_id    = getattr(det, 'id', '')
                frame_id  = getattr(det, 'frame_id', '') or getattr(getattr(det, 'bbox3d', None), 'frame_id', '') \
                            or getattr(getattr(msg, 'header', None), 'frame_id', '')

                # 2D bbox (pixels)
                bbox2d    = getattr(det, 'bbox', None)
                cx_px = cy_px = w_px = h_px = float('nan')
                if bbox2d:
                    center2d   = getattr(bbox2d, 'center', None)
                    size2d     = getattr(bbox2d, 'size', None)
                    if center2d:
                        pos2d   = getattr(center2d, 'position', center2d)  # handle Pose2D vs Point2D
                        cx_px   = float(getattr(pos2d, 'x', float('nan')))
                        cy_px   = float(getattr(pos2d, 'y', float('nan')))
                    if size2d:
                        w_px    = float(getattr(size2d, 'x', float('nan')))
                        h_px    = float(getattr(size2d, 'y', float('nan')))

                # 3D bbox center (meters) + size (meters)
                bbox3d    = getattr(det, 'bbox3d', None)
                x = y = z = dx = dy = dz = float('nan')
                if bbox3d:
                    center3d = getattr(bbox3d, 'center', None)
                    size3d   = getattr(bbox3d, 'size', None)
                    if center3d:
                        pos3d = getattr(center3d, 'position', None)
                        if pos3d:
                            x = float(getattr(pos3d, 'x', float('nan')))
                            y = float(getattr(pos3d, 'y', float('nan')))
                            z = float(getattr(pos3d, 'z', float('nan')))
                    if size3d:
                        dx = float(getattr(size3d, 'x', float('nan')))
                        dy = float(getattr(size3d, 'y', float('nan')))
                        dz = float(getattr(size3d, 'z', float('nan')))

                # Range & angles (optical frame)
                rng   = math.sqrt(x*x + y*y + z*z) if all(map(lambda v: not math.isnan(v), [x,y,z])) else float('nan')
                az    = deg(math.atan2(x, z)) if (not math.isnan(x) and not math.isnan(z)) else float('nan')
                el    = deg(math.atan2(-y, z)) if (not math.isnan(y) and not math.isnan(z)) else float('nan')

                # Mask (2D)
                mask   = getattr(det, 'mask', None)
                mask_h = int(getattr(mask, 'height', 0)) if mask else 0
                mask_w = int(getattr(mask, 'width', 0))  if mask else 0

                # Print one concise line per detection
                self.get_logger().info(
                    f"[{i}] id={det_id} class='{cls_name}'({cls_id}) conf={score:.3f} | "
                    f"2Dcx={cx_px:.1f}px 2Dcy={cy_px:.1f}px w={w_px:.1f}px h={h_px:.1f}px | "
                    f"3D=(x={x:.3f}, y={y:.3f}, z={z:.3f}) m  size=(dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}) m | "
                    f"z_fwd={z:.3f} m  range={rng:.3f} m  az={az:.1f}°  el={el:.1f}° | "
                    f"mask={mask_h}x{mask_w}  frame='{frame_id}'"
                )

        # vision_msgs layout (fallback)
        else:
            for i, det in enumerate(getattr(msg, 'detections', [])):
                # results -> hypothesis.id/score
                label = score = None
                results = getattr(det, 'results', [])
                if results:
                    hyp = getattr(results[0], 'hypothesis', results[0])
                    label = getattr(hyp, 'class_id', getattr(hyp, 'id', ''))
                    score = float(getattr(hyp, 'score', 0.0))

                bbox3d = getattr(det, 'bbox', None)  # vision_msgs calls it bbox (3D)
                x = y = z = dx = dy = dz = float('nan')
                if bbox3d:
                    center = getattr(bbox3d, 'center', None)
                    size   = getattr(bbox3d, 'size', None)
                    if center:
                        pos = getattr(center, 'position', None)
                        if pos:
                            x = float(pos.x); y = float(pos.y); z = float(pos.z)
                    if size:
                        dx = float(size.x); dy = float(size.y); dz = float(size.z)

                rng = math.sqrt(x*x + y*y + z*z) if all(map(lambda v: not math.isnan(v), [x,y,z])) else float('nan')
                az  = deg(math.atan2(x, z)) if (not math.isnan(x) and not math.isnan(z)) else float('nan')
                el  = deg(math.atan2(-y, z)) if (not math.isnan(y) and not math.isnan(z)) else float('nan')

                frame_id = getattr(getattr(msg, 'header', None), 'frame_id', '')

                self.get_logger().info(
                    f"[{i}] class='{label}' conf={score:.3f} | "
                    f"3D=(x={x:.3f}, y={y:.3f}, z={z:.3f}) m  size=(dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}) m | "
                    f"z_fwd={z:.3f} m  range={rng:.3f} m  az={az:.1f}°  el={el:.1f}° | "
                    f"frame='{frame_id}'"
                )

def main():
    parser = argparse.ArgumentParser(description="Print YOLO 3D detections with distances and angles.")
    parser.add_argument("--topic", default="/yolo/detections_3d", help="Detections 3D topic")
    args = parser.parse_args()

    rclpy.init()
    node = Yolo3DInspector(args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

