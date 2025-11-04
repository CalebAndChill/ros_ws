# Copyright (C) 2023 Miguel Ángel González Santamarta
# GPLv3 license

from typing import List, Dict, Optional
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

from rcl_interfaces.msg import SetParametersResult, ParameterType
import yaml
import numpy as np

import torch
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

# Try to import the visual-prompt predictor; if unavailable, we'll warn at runtime.
try:
    # Common import path in recent Ultralytics
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # type: ignore
except Exception:  # pragma: no cover
    YOLOEVPSegPredictor = None  # type: ignore

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # ------------------- Base params -------------------
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        # ------------------- TEXT PROMPTS -------------------
        # Force STRING_ARRAY by giving a string list default; we strip "__dummy__" later.
        self.declare_parameter("classes", ["__dummy__"])

        # ------------------- VISUAL (PICTURE) PROMPTS -------------------
        # Toggle + inputs. bboxes and cls are parsed from YAML strings or string arrays.
        self.declare_parameter("vp_enable", False)
        self.declare_parameter("vp_refer_image", "")
        self.declare_parameter("vp_bboxes", "[]")           # e.g., "[[x1,y1,x2,y2],[...]]"
        self.declare_parameter("vp_cls", "[]")               # e.g., "[0,1,...]" (sequential IDs)
        self.declare_parameter("vp_use_current_frame", False)

        # YOLO class map
        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}

        # runtime prompt state
        self.classes: List[str] = []
        self.vp_enable: bool = False
        self.vp_refer_image: str = ""
        self.vp_use_current_frame: bool = False
        self.vp_bboxes: Optional[np.ndarray] = None
        self.vp_cls: Optional[np.ndarray] = None
        self._vp_seeded: bool = False  # set true once we seed from current frame

        # live param updates
        self.add_on_set_parameters_callback(self._on_params_changed)

    # ------------------- Lifecycle -------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )

        # ros params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        # ----- parse 'classes' (strip "__dummy__"); accepts string[] or YAML string -----
        self.classes = []
        try:
            pv = self.get_parameter("classes").get_parameter_value()
            if pv.type == ParameterType.PARAMETER_STRING_ARRAY:
                lst = list(pv.string_array_value)
                self.classes = [s for s in lst if s and s != "__dummy__"]
            elif pv.type == ParameterType.PARAMETER_STRING:
                s = (pv.string_value or "").strip()
                if s:
                    parsed = list(yaml.safe_load(s))
                    self.classes = [x for x in parsed if x and x != "__dummy__"]
        except Exception as e:
            self.get_logger().warn(f"Failed to parse 'classes' param: {e}")

        # ----- parse visual-prompt parameters -----
        self._parse_visual_prompt_params_from_server()

        # publishers/subscribers QoS
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        # YOLOE does not support fusing
        if isinstance(self.yolo, YOLO) or isinstance(self.yolo, YOLOWorld):
            try:
                self.get_logger().info("Trying to fuse model...")
                self.yolo.fuse()
            except TypeError as e:
                self.get_logger().warn(f"Error while fuse: {e}")

        # services
        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        # expose SetClasses for YOLOWorld **and** YOLOE
        if isinstance(self.yolo, (YOLOWorld, YOLOE)):
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )

        # subscriber
        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb, self.image_qos_profile
        )

        # Apply text prompts if provided
        try:
            if isinstance(self.yolo, (YOLOWorld, YOLOE)) and self.classes:
                self.yolo.set_classes(self.classes)  # YOLOE needs CLIP installed
                self.get_logger().info(f"Open-vocab classes set: {self.classes}")
        except Exception as e:
            self.get_logger().warn(f"set_classes failed: {e}")

        # Warn if user enabled VP but predictor is missing
        if self.vp_enable and YOLOEVPSegPredictor is None:
            self.get_logger().warn(
                "vp_enable is True, but YOLOEVPSegPredictor could not be imported. "
                "Please update Ultralytics or adjust the import path."
            )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        if hasattr(self, "_set_classes_srv") and self._set_classes_srv is not None:
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    # ------------------- Services -------------------

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        self.get_logger().info(f"Setting classes: {req.classes}")
        try:
            if isinstance(self.yolo, (YOLOWorld, YOLOE)):
                self.yolo.set_classes(list(req.classes))
                self.classes = list(req.classes)
                self.get_logger().info(f"New classes: {self.yolo.names}")
            else:
                self.get_logger().warn(
                    "SetClasses called, but model does not support open-vocab prompts."
                )
        except Exception as e:
            self.get_logger().warn(f"set_classes failed: {e}")
        return res

    # ------------------- Helpers (publish formatting) -------------------

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []
        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list = []
        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                msg = BoundingBox2D()
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])
                boxes_list.append(msg)
        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])
                boxes_list.append(msg)
        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks or []:
            msg = Mask()
            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]
            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        keypoints_list = []
        points: Keypoints
        for points in results.keypoints or []:
            msg_array = KeyPoint2DArray()
            if points.conf is None:
                continue
            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self.threshold:
                    msg = KeyPoint2D()
                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)
                    msg_array.data.append(msg)
            keypoints_list.append(msg_array)
        return keypoints_list

    # ------------------- Main callback -------------------

    def image_cb(self, msg: Image) -> None:
        if not self.enable:
            return

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.yolo_encoding)

        # --- Visual-prompt kwargs (optional) ---
        vp_kwargs = {}
        if (
            self.vp_enable
            and YOLOEVPSegPredictor is not None
            and self.vp_bboxes is not None
            and self.vp_cls is not None
            and len(self.vp_bboxes) > 0
            and len(self.vp_cls) > 0
        ):
            # choose a reference image (path or current frame)
            refer: Optional[object] = None
            if self.vp_refer_image:
                refer = self.vp_refer_image  # path string is accepted by Ultralytics
            elif self.vp_use_current_frame:
                refer = cv_image

            vp_kwargs.update(
                {
                    "visual_prompts": {"bboxes": self.vp_bboxes, "cls": self.vp_cls},
                    "predictor": YOLOEVPSegPredictor,
                }
            )
            if refer is not None:
                vp_kwargs["refer_image"] = refer
                # Mark we have seeded once from the stream; you can keep sending prompts
                # if you want to update them, but it's not required every frame.
                if self.vp_use_current_frame and not self._vp_seeded:
                    self._vp_seeded = True

        # inference
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            device=self.device,
            **vp_kwargs,
        )
        results: Results = results[0].cpu()

        # parse
        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)
        else:
            hypothesis, boxes = [], []

        masks = self.parse_masks(results) if results.masks else []
        keypoints = self.parse_keypoints(results) if results.keypoints else []

        # publish
        detections_msg = DetectionArray()
        for i in range(len(results)):
            aux_msg = Detection()

            if (results.boxes or results.obb) and hypothesis and boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

            if results.masks and masks:
                aux_msg.mask = masks[i]

            if results.keypoints and keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        detections_msg.header = msg.header
        self._pub.publish(detections_msg)

        # cleanup
        del results
        del cv_image

    # ------------------- Param updates -------------------

    def _on_params_changed(self, params):
        """Allow live updates to 'classes' and visual-prompt params."""
        new_classes = None
        vp_changed = False

        for p in params:
            # ----- classes (text) -----
            if p.name == "classes":
                try:
                    if p.type_ == ParameterType.PARAMETER_STRING_ARRAY:
                        lst = list(p.value)
                        new_classes = [s for s in lst if s and s != "__dummy__"]
                    elif p.type_ == ParameterType.PARAMETER_STRING:
                        s = (p.value or "").strip()
                        parsed = list(yaml.safe_load(s)) if s else []
                        new_classes = [x for x in parsed if x and x != "__dummy__"]
                except Exception as e:
                    self.get_logger().warn(f"Failed to parse updated 'classes': {e}")

            # ----- visual prompts -----
            elif p.name == "vp_enable":
                self.vp_enable = bool(p.value)
                vp_changed = True
            elif p.name == "vp_refer_image":
                self.vp_refer_image = str(p.value or "")
                vp_changed = True
            elif p.name == "vp_use_current_frame":
                self.vp_use_current_frame = bool(p.value)
                vp_changed = True
            elif p.name in ("vp_bboxes", "vp_cls"):
                try:
                    # Accept YAML string or array-like
                    parsed = None
                    if p.type_ == ParameterType.PARAMETER_STRING and p.value:
                        parsed = yaml.safe_load(str(p.value))
                    elif p.type_ == ParameterType.PARAMETER_STRING_ARRAY and p.value:
                        parsed = list(p.value)
                    if p.name == "vp_bboxes":
                        self.vp_bboxes = (
                            np.array(parsed, dtype=float) if parsed is not None else None
                        )
                    else:
                        self.vp_cls = (
                            np.array(parsed, dtype=int) if parsed is not None else None
                        )
                    vp_changed = True
                except Exception as e:
                    self.get_logger().warn(f"Failed to parse '{p.name}': {e}")

        # apply classes immediately if model supports it
        if new_classes is not None:
            try:
                if hasattr(self, "yolo") and isinstance(self.yolo, (YOLOWorld, YOLOE)):
                    self.yolo.set_classes(new_classes)
                    self.classes = new_classes
                    self.get_logger().info(
                        f"Updated open-vocab classes: {self.classes}"
                    )
                else:
                    # Will be applied on activate
                    self.classes = new_classes
                    self.get_logger().info(
                        f"Queued open-vocab classes for activation: {self.classes}"
                    )
            except Exception as e:
                self.get_logger().warn(f"set_classes (update) failed: {e}")

        if vp_changed and self.vp_enable and YOLOEVPSegPredictor is None:
            self.get_logger().warn(
                "Visual prompts updated, but YOLOEVPSegPredictor import failed. "
                "Please ensure Ultralytics supports VP in your install."
            )

        return SetParametersResult(successful=True)

    # ------------------- Internal helpers -------------------

    def _parse_visual_prompt_params_from_server(self) -> None:
        """Parse visual-prompt parameters from node parameters."""
        try:
            self.vp_enable = self.get_parameter("vp_enable").get_parameter_value().bool_value
            self.vp_refer_image = self.get_parameter("vp_refer_image").get_parameter_value().string_value
            self.vp_use_current_frame = (
                self.get_parameter("vp_use_current_frame").get_parameter_value().bool_value
            )

            def _parse_list_param(name: str):
                pv = self.get_parameter(name).get_parameter_value()
                # Try string first (YAML), then string array
                if pv.type == ParameterType.PARAMETER_STRING:
                    s = (pv.string_value or "").strip()
                    if s:
                        return list(yaml.safe_load(s))
                if pv.type == ParameterType.PARAMETER_STRING_ARRAY:
                    return list(pv.string_array_value)
                return []

            vp_b = _parse_list_param("vp_bboxes")
            vp_c = _parse_list_param("vp_cls")

            self.vp_bboxes = np.array(vp_b, dtype=float) if len(vp_b) else None
            self.vp_cls = np.array(vp_c, dtype=int) if len(vp_c) else None

            self._vp_seeded = False

        except Exception as e:
            self.get_logger().warn(f"Failed to parse visual-prompt params: {e}")


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
