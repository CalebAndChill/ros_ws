#!/usr/bin/env python3
import os, cv2, json, argparse, time
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

HELP_TEXT = "Keys: [c]=capture+annotate  [w]=capture only  [q]=quit  [h]=help"

def parse_size(s):
    if not s: return None
    try:
        w,h = s.lower().split('x')
        return (int(w), int(h))
    except Exception:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, e.g. 960x544")

def make_qos(which: str):
    which = (which or "best_effort").lower()
    rel = QoSReliabilityPolicy.BEST_EFFORT if which.startswith('best') else QoSReliabilityPolicy.RELIABLE
    return QoSProfile(
        reliability=rel,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=1,
    )

class CamPreview(Node):
    def __init__(self, topic: str, qos: str):
        super().__init__('cam_capture_annotate')
        self.bridge = CvBridge()
        self.last = None
        self.sub = self.create_subscription(Image, topic, self.cb, make_qos(qos))
        self.get_logger().info(f"Subscribing to {topic} (QoS={qos})")

    def cb(self, msg: Image):
        try:
            self.last = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")

def draw_text(img, text, y=28):
    cv2.rectangle(img, (0,0), (img.shape[1], 40), (0,0,0), -1)
    cv2.putText(img, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def save_image(img, out_dir, resize_to):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f"ref_{ts}.png")
    to_write = img
    if resize_to is not None:
        to_write = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, to_write)
    return path, to_write.shape[1], to_write.shape[0]  # (path, W, H)

def annotate_and_print(img, ref_path, default_cls=0):
    # Let user draw one or more boxes; ENTER to accept each, ESC/ENTER to finish
    rois = cv2.selectROIs("Draw boxes (ENTER accept, ESC finish)", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if rois is None or len(rois) == 0:
        print("No boxes selected.")
        return

    bboxes, cls = [], []
    last = default_cls
    H, W = img.shape[:2]
    for i,(x,y,w,h) in enumerate(rois):
        x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
        # clamp to image
        x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
        y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping degenerate ROI #{i}: {(x1,y1,x2,y2)}"); continue
        bboxes.append([x1,y1,x2,y2])
        try:
            c = input(f"Class id for box #{i} (default {last}): ").strip()
            c = int(c) if c != "" else last
        except:
            c = last
        cls.append(c); last = c

    bb_yaml = json.dumps(bboxes)
    cls_yaml = json.dumps(cls)
    # Print ready-to-paste
    print("\nCopy into your launch command:")
    print(f"vp_refer_image:={ref_path}")
    print(f"vp_bboxes:='{bb_yaml}'")
    print(f"vp_cls:='{cls_yaml}'")
    # Save a sidecar text file next to the image
    with open(os.path.splitext(ref_path)[0] + "_vp.txt", "w") as f:
        f.write(f"vp_refer_image:={ref_path}\nvp_bboxes:='{bb_yaml}'\nvp_cls:='{cls_yaml}'\n")
    print(f"Wrote {os.path.splitext(ref_path)[0] + '_vp.txt'}")

def main():
    ap = argparse.ArgumentParser(description="Live preview → capture → (optional) annotate for YOLO-E visual prompts")
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--qos", choices=["best_effort","reliable"], default="best_effort")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/Pictures"))
    ap.add_argument("--resize", type=parse_size, default=None, help="Optional save resize, e.g. 960x544")
    ap.add_argument("--class-id", type=int, default=0, help="Default class id when annotating")
    args = ap.parse_args()

    rclpy.init()
    node = CamPreview(args.image_topic, args.qos)

    win = "Camera Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    print(HELP_TEXT)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            frame = node.last
            if frame is None:
                blank = 480, 640, 3
                canvas = (255 * (0.1 + 0.0)).__class__
                img = (255 * 0.1) * (0).__class__
                img = 255 * 0.1
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)
                # just show a waiting screen
                img = 255 * (0.1) * (0).__class__
                img = (255 * 0.1)
            if frame is None:
                img = 255 * (0.1) * (0).__class__

            if frame is None:
                img = 255 * (0.1)
            if frame is None:
                img = 255 * (0.1)
            if frame is None:
                img = 255 * (0.1)
            # Create a gray waiting frame
            if frame is None:
                img = (255 * 0.1) * (0).__class__
            if frame is None:
                img = (255 * 0.1)
            if frame is None:
                img = (255 * 0.1)

            if frame is None:
                img = (255 * 0.1)
                img = (255 * 0.1)
                img = (255 * 0.1)

            if frame is None:
                img = (255 * 0.1)
                img = (255 * 0.1)
                # Use a simple image
                img = (255 * 0.1)
                img = (255 * 0.1)
            if frame is None:
                img = (255 * 0.1)
                # fallback plain
                img = (255 * 0.1)
                # actual plain gray image:
                import numpy as np
                img = (np.ones((480, 640, 3), dtype=np.uint8) * 40)
                draw_text(img, f"Waiting for {args.image_topic} ...  {HELP_TEXT}")
            else:
                img = frame.copy()
                H, W = img.shape[:2]
                draw_text(img, f"{W}x{H}  |  {HELP_TEXT}")

            cv2.imshow(win, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):  # q or ESC
                break
            elif k == ord('h'):
                print(HELP_TEXT)
            elif k == ord('w') and node.last is not None:
                path, W, H = save_image(node.last, args.out_dir, args.resize)
                print(f"Saved {path} at {W}x{H}  (no annotation)")
            elif k == ord('c') and node.last is not None:
                # Save first (so annotation file is next to it), then annotate that saved image
                path, W, H = save_image(node.last, args.out_dir, args.resize)
                print(f"Saved {path} at {W}x{H}  → annotate now")
                img_to_annotate = cv2.imread(path)
                # Wayland tip: if window controls are unresponsive, run with: QT_QPA_PLATFORM=xcb
                annotate_and_print(img_to_annotate, path, default_cls=args.class_id)

    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
