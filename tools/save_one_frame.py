import rclpy, cv2, os
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

TOPIC = "/camera/camera/color/image_raw"
OUT   = "/home/caleb/Pictures/ref_from_cam.png"

class SaveOne(Node):
    def __init__(self):
        super().__init__("save_one_frame")
        self.bridge = CvBridge()
        q = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.sub = self.create_subscription(Image, TOPIC, self.cb, q)
        self.get_logger().info(f"Waiting for first frame on {TOPIC} ...")

    def cb(self, msg: Image):
        self.destroy_subscription(self.sub)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        cv2.imwrite(OUT, img)
        h, w = img.shape[:2]
        self.get_logger().info(f"Saved {OUT} at {w}x{h}")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = SaveOne()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
