#!/usr/bin/env python3
import os, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

TOPIC = "/camera/camera/color/image_raw"
OUT_DIR = "/home/caleb/Pictures/Dataset"

class Capture(Node):
    def __init__(self):
        super().__init__("capture_on_key")
        self.bridge = CvBridge()
        self.last = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Image, TOPIC, self.cb, qos)
        os.makedirs(OUT_DIR, exist_ok=True)
        self.get_logger().info(f"Previewing {TOPIC}. Press 's' to save, 'q' to quit. Saving to {OUT_DIR}")

    def cb(self, msg: Image):
        self.last = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def main():
    rclpy.init()
    node = Capture()

    cv2.namedWindow("ROS Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROS Capture", 960, 540)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)

            if node.last is None:
                img = np.ones((480, 640, 3), dtype=np.uint8) * 30
                cv2.putText(img, "Waiting for frames...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            else:
                img = node.last.copy()
                cv2.putText(img, "s=save  q=quit", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            cv2.imshow("ROS Capture", img)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("q") or k == 27:
                break
            if k == ord("s") and node.last is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(OUT_DIR, f"cap_{ts}.png")
                cv2.imwrite(path, node.last)
                print("Saved:", path)

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()