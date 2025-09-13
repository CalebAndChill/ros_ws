# Copyright (C) 2023 Miguel Ángel González Santamarta
# GPLv3

import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("yolo_bringup"),
                        "launch",
                        "yolo.launch.py",
                    )
                ),
                launch_arguments={
                    # lock to YOLO-E
                    "model_type": "YOLOE",

                    # NOTE: non-PF so text prompts work
                    "model": LaunchConfiguration("model", default="yoloe-11l-seg.pt"),

                    "tracker": LaunchConfiguration("tracker", default="bytetrack.yaml"),
                    "device": LaunchConfiguration("device", default="cuda:0"),
                    "enable": LaunchConfiguration("enable", default="True"),
                    "threshold": LaunchConfiguration("threshold", default="0.5"),
                    "input_image_topic": LaunchConfiguration(
                        "input_image_topic", default="/camera/rgb/image_raw"
                    ),
                    "image_reliability": LaunchConfiguration(
                        "image_reliability", default="1"
                    ),
                    "namespace": LaunchConfiguration("namespace", default="yolo"),

                    # NEW: forward text prompts to yolo.launch.py → yolo_node.py
                    # You can pass it as a YAML list: classes:="['mug','bottle']"
                    "classes": LaunchConfiguration("classes", default="['__dummy__']"),

                    # (optional) expose tracking/3D toggles via this wrapper too:
                    # "use_tracking": LaunchConfiguration("use_tracking", default="True"),
                    # "use_3d": LaunchConfiguration("use_3d", default="False"),
                }.items(),
            )
        ]
    )
