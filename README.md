# ros2_ws

Videos (2X speed)
-------------------------------------------------------------------------------------------------------------

https://github.com/user-attachments/assets/09ae672a-cc2a-4a10-a6eb-ea4afa5b48b1

https://github.com/user-attachments/assets/d4d66ab9-284d-49f1-907c-75693aaede99

https://github.com/user-attachments/assets/646a4d0f-3ab2-48c5-a87e-5cd1329bee5c


# G1 Vision-Guided Manipulation Pipeline

A vision-guided robotic manipulation stack for the **Unitree G1 EDU** (29-DOF humanoid) with **Dex3-1** three-finger hands and an **Intel RealSense D435i** RGB-D camera.

## The pipeline:

<img width="2033" height="1012" alt="Flowchart Whiteboard in Bright Green Lime Green Pink Corporate Neon Style (15)" src="https://github.com/user-attachments/assets/e89ee6cf-4da6-4f9f-a109-0efd74c6d3c6" />

## Table of Contents

1. [Hardware](#hardware)
2. [Dependencies](#dependencies)
3. [Setup](#setup)
4. [Models](#models)
5. [Network & sourcing](#network--sourcing)
6. [Main pipeline files](#main-pipeline-files)
   - [save_one_frame.py](#save_one_framepy)
   - [vp_box_tool.py](#vp_box_toolpy)
   - [yolo.launch.py / yoloe.launch.py — Detection](#yololaunchpy--yoloelaunchpy--detection)
   - [best_mask_pca.py](#best_mask_pcapy)
   - [pca_obb_to_torso_pregrasp.py](#pca_obb_to_torso_pregrasppy)
   - [g1_ik_ready_then_target_dex3_table.py](#g1_ik_ready_then_target_dex3_tablepy)
7. [End-to-end demo](#end-to-end-demo)
8. [Simulation (optional)](#simulation-optional)
9. [Troubleshooting](#troubleshooting)

---

## Hardware

| Component | Notes |
|---|---|
| Unitree G1 EDU | 29-DOF humanoid. Joint indices: legs 0–11, waist 12–14, left arm 15–21, right arm 22–28, weight slot 29. |
| Dex3-1 three-finger hand (right) | DDS topic `rt/dex3/right/cmd`; state on `rt/dex3/right/state` or `rt/lf/dex3/right/state`. |
| Intel RealSense D435i | USB to host laptop. URDF frame `d435_optical_frame` ≡ driver frame `camera_color_optical_frame`. |
| USB-C ethernet adapter | Interface `enx00e04c7015d3` to the robot. |

---

## Dependencies

### System
- **Ubuntu 22.04**
- **ROS 2 Humble** (desktop install)
- **CycloneDDS** (`rmw_cyclonedds_cpp`)

### ROS 2 packages (system)

```bash
sudo apt install \
  ros-humble-realsense2-camera \
  ros-humble-cv-bridge \
  ros-humble-sensor-msgs-py \
  ros-humble-tf2-ros \
  ros-humble-vision-msgs
```

### yolo_ros (provides `yolo_bringup` and `yolo_msgs`)

Tested with [`mgonzs13/yolo_ros`](https://github.com/mgonzs13/yolo_ros) **v4.6.1+**. The upstream project now uses [`uv`](https://github.com/astral-sh/uv) to manage its Python dependencies (Ultralytics, PyTorch, etc.) in an isolated environment — follow their install verbatim:

```bash
# Clone into your colcon src
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git

# Install uv and let it resolve yolo_ros' Python deps
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
cd ~/ros2_ws/src/yolo_ros
uv sync

# rosdep + colcon
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

`uv sync` keeps Ultralytics / PyTorch confined to yolo_ros' own venv — this is what fixed the old NumPy-2-breaks-cv_bridge issue. **Do not `pip install ultralytics` at system level**; let uv handle it.

Compatible models include YOLOv3 through YOLOv12, YOLOv26, YOLO-World, and YOLOE.

### Python (system — for the standalone scripts in this repo)

The scripts under `~/ros2_ws/tools/` run via `python3` directly (not through uv), so they need their own system-Python packages. None of these scripts import `ultralytics` or `torch` — those live inside yolo_ros' uv venv.

```bash
pip install \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "pin" \
  "scipy" \
  "huggingface_hub"
```

- `numpy==1.26.4` — pinned defensively. Anything pulling NumPy 2.x at system level still breaks `cv_bridge`. The uv migration in yolo_ros removes the most common source of this drift, but a stray `pip install` of something else can still trigger it.
- `opencv-python==4.10.0.84` — pinned for compatibility with the pinned NumPy.
- `pin` = Pinocchio (FK / IK).
- `scipy` — used by `best_mask_pca.py` for connected-component filtering.
- `huggingface_hub` — only needed if you use the Python download path for `best2.pt`. Skip it if you use `wget`.

### Unitree SDK2 (real-robot only)
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
python3 -m venv ~/unitree_sdk2_venv
source ~/unitree_sdk2_venv/bin/activate
cd ~/unitree_sdk2_python && pip install -e .
deactivate
```

The IK / SDK scripts add `~/unitree_sdk2_python` and the venv's `site-packages` to `sys.path` automatically — no need to activate the venv manually.

### URDF
```
/home/<user>/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf
https://github.com/isri-aist/g1_description/blob/main/urdf/g1_29dof.urdf 
```
---

## Setup

Once the dependencies above are in place:

```bash
# 1. (If you haven't already) clone & build yolo_ros — see Dependencies above
# 2. Build your own colcon workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash

# 3. Fetch the custom-trained model
bash scripts/download_models.sh
```

Verify the install:

```bash
ros2 pkg list | grep yolo            # should show yolo_bringup and yolo_msgs
ros2 launch yolo_bringup yolo.launch.py --show-args   # should print the parameter list
python3 -c "import pinocchio; print(pinocchio.__version__)"
python3 -c "import cv_bridge; import numpy; print('cv_bridge OK, numpy', numpy.__version__)"
```

If the last command crashes, you've got a NumPy 2.x leak somewhere in your system Python — re-pin (`pip install "numpy==1.26.4" "opencv-python==4.10.0.84"`).

---

## Models

The pipeline can run with one of four model paths:

| Model | Source | Use |
|---|---|---|
| `best2.pt` | **This repo's Hugging Face mirror** — [`calebandchill/cerealbox.torch.toothpaste`](https://huggingface.co/calebandchill/cerealbox.torch.toothpaste) | Custom-trained YOLO seg model for **cereal box, torch, toothpaste** |
| `yolo11s-seg.pt` | Ultralytics (auto-downloads) | Vanilla YOLO11 segmentation, trained on COCO 80 classes (bottle, cup, mouse, keyboard, book, scissors, etc.) |
| `yoloe-11s-seg.pt` | Ultralytics | YOLOE with **text prompts** (open vocabulary, any class via `classes:=[...]`). Also supports **picture prompts** (visual reference). |
| `yoloe-11s-seg-pf.pt` | Ultralytics | YOLOE **prompt-free** — no class list needed, returns what it sees. Useful as a sanity check or fallback if the MobileCLIP text encoder is broken locally. |

### Downloading `best2.pt` from Hugging Face

Direct URL — no auth needed (public model):

```bash
mkdir -p ~/ros2_ws/Models
wget -nc \
  https://huggingface.co/calebandchill/cerealbox.torch.toothpaste/resolve/main/best2.pt \
  -O ~/ros2_ws/Models/best2.pt
```

Or via `huggingface_hub` (caches in `~/.cache/huggingface/`, instant on re-runs):

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="calebandchill/cerealbox.torch.toothpaste",
    filename="best2.pt",
    local_dir="/home/caleb/ros2_ws/Models",
)
```

The YOLOE checkpoints auto-download to Ultralytics' cache the first time you reference them.

---

## Network & sourcing

Add to `~/.bashrc`:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="0"><General>
  <NetworkInterfaceAddress>enx00e04c7015d3</NetworkInterfaceAddress>
</General></Domain></CycloneDDS>'

alias sros='source /opt/ros/humble/setup.bash && \
            source ~/unitree_ros2/setup.sh && \
            source ~/ros2_ws/install/setup.bash && \
            ros2 daemon stop && ros2 daemon start'
```

`unitree_ros2/setup.sh` MUST be sourced **before** `ros2_ws/install/setup.bash`. A daemon started before sourcing silently breaks topic discovery — always `daemon stop && daemon start` after sourcing.

---

## Main pipeline files

Each section below covers what the file does, where it lives, and the parameters worth knowing.

---

### `save_one_frame.py`

Saves one color frame from `/camera/camera/color/image_raw` to PNG. Used to capture a reference image for YOLOE visual-prompt mode.

**Location:** `~/ros2_ws/tools/save_one_frame.py`

**Run:**
```bash
python3 ~/ros2_ws/tools/save_one_frame.py
```

**Parameters (edit at top of file):**

| Constant | Default | Description |
|---|---|---|
| `TOPIC` | `/camera/camera/color/image_raw` | Color image topic. Change for the sim path (`/d435i/image`). |
| `OUT` | `~/Pictures/ref_from_cam.png` | Output path. |

---

### `vp_box_tool.py`

Interactive OpenCV ROI tool. Draw one or more bounding boxes on a saved frame, get back the `vp_bboxes:='[[...]]'` and `vp_cls:='[...]'` YAML strings to paste into the YOLOE launch command.

**Location:** `~/ros2_ws/tools/vp_box_tool.py`

**Run:**
```bash
python3 ~/ros2_ws/tools/vp_box_tool.py ~/Pictures/ref_from_cam.png [class_id]
```

- Click-drag a box, ENTER to accept, ESC to finish.
- Output is printed to stdout and written to a `_vp.txt` file alongside the image.

---

### `yolo.launch.py` / `yoloe.launch.py` — Detection

Wrapper around `yolo_bringup` that runs YOLO / YOLOE / YOLO-World against the live RealSense feed and (optionally) the aligned depth image. Four operating modes — pick the one that matches your model:

| Mode | Launch file | Model | Class source |
|---|---|---|---|
| (A) Vanilla YOLO | `yolo.launch.py` | `yolo11s-seg.pt` | COCO 80 (built in) |
| (B) Custom-trained YOLO | `yolo.launch.py` | `best2.pt` | Whatever the model was trained on (cereal box / torch / toothpaste) |
| (C) YOLOE text-prompt | `yoloe.launch.py` | `yoloe-11s-seg.pt` | `classes:="['mug','bottle',...]"` |
| (D) YOLOE picture-prompt | `yoloe.launch.py` | `yoloe-11s-seg.pt` | Bounding box on a reference image |

**Location:** `~/ros2_ws/src/yolo_ros/yolo_bringup/launch/{yolo,yoloe}.launch.py`

#### (A) Vanilla YOLO — COCO classes, 2D only

Quick smoke test, no depth needed:

```bash
ros2 launch yolo_bringup yolo.launch.py \
  model:=/home/$USER/ros2_ws/Models/yolo11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  yolo_encoding:=rgb8 \
  imgsz_width:=960 imgsz_height:=544 \
  use_3d:=False device:=cpu
```

Same model with 3D enabled (publishes `/yolo/detections_3d`, which `best_mask_pca.py` consumes):

```bash
ros2 launch yolo_bringup yolo.launch.py \
  model:=/home/$USER/ros2_ws/Models/yolo11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 \
  imgsz_width:=960 imgsz_height:=544 \
  depth_image_units_divisor:=1000 \
  maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame \
  use_3d:=True device:=cpu
```

Available COCO classes (the 80 things `yolo11s-seg.pt` will detect): `person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush`.

#### (B) Custom-trained model (`best2.pt`)

This one detects **cereal box, torch (small flashlight), toothpaste** — the classes the model was trained on. Don't pass `classes:=...`; it's not a YOLOE model.

```bash
ros2 launch yolo_bringup yolo.launch.py \
  model:=/home/$USER/ros2_ws/Models/best2.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 imgsz_width:=960 imgsz_height:=544 \
  depth_image_units_divisor:=1000 \
  maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame \
  use_3d:=True device:=cpu
```

#### (C) YOLOE text-prompt — open vocabulary

Pass arbitrary class names as a YAML list. The model produces masks for whatever text prompts you give it (within reason — bizarre prompts return nothing).

2D only:
```bash
ros2 launch yolo_bringup yoloe.launch.py \
  model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  device:=cpu use_3d:=False use_tracking:=False \
  threshold:=0.10 \
  classes:="['mug','bottle','keyboard','mouse','phone','hammer']"
```

Full 3D pipeline:
```bash
ros2 launch yolo_bringup yolo.launch.py \
  model_type:=YOLOE \
  model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 imgsz_width:=960 imgsz_height:=544 \
  threshold:=0.10 \
  classes:="['mug','bottle','keyboard','mouse','phone','hammer','apple']" \
  depth_image_units_divisor:=1000 maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame \
  use_3d:=True device:=cpu
```

**Prompt-free variant** (`yoloe-11s-seg-pf.pt`) — no `classes:=` needed, returns whatever it sees. Useful when the text encoder is broken or you just want to see what's in the scene:

```bash
ros2 launch yolo_bringup yoloe.launch.py \
  model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg-pf.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  device:=cpu use_3d:=False
```

#### (D) YOLOE picture-prompt — visual reference

Instead of text classes, give YOLOE a **reference image with a bounding box** around an example of the object. The model then finds visually similar things in the live feed. Useful for objects that don't have a clean text name (a specific toy, a custom-made part, a tangled cable).

Three-step workflow:

1. **Capture a reference frame** from the camera:
   ```bash
   python3 ~/ros2_ws/tools/save_one_frame.py
   # writes ~/Pictures/ref_from_cam.png
   ```

2. **Draw a bounding box** around the object of interest using the OpenCV ROI tool:
   ```bash
   python3 ~/ros2_ws/tools/vp_box_tool.py ~/Pictures/ref_from_cam.png
   # click-drag a box, ENTER to accept, ESC to finish
   # prints vp_bboxes:='[[x1,y1,x2,y2]]' and vp_cls:='[0]' to copy
   ```

3. **Launch YOLOE in visual-prompt mode** with the printed strings:
   ```bash
   ros2 launch yolo_bringup yoloe.launch.py \
     model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg.pt \
     input_image_topic:=/camera/camera/color/image_raw \
     device:=cpu use_3d:=False \
     vp_enable:=True \
     vp_refer_image:=/home/$USER/Pictures/ref_from_cam.png \
     vp_bboxes:='[[221,405,345,858]]' \
     vp_cls:='[0]' \
     threshold:=0.005
   ```

You almost always need to drop `threshold` very low (0.005–0.05) for picture-prompt mode — confidence scores are much lower than text-prompt mode.

#### Key parameters (all modes)

| Argument | Default | What it does / when to change |
|---|---|---|
| `model` | `yolov8m.pt` | Path to .pt file. |
| `model_type` | `YOLO` | Set `YOLOE` when using a YOLOE checkpoint, `World` for YOLO-World. |
| `classes` | `['__dummy__']` | Text prompts for YOLOE (mode C). Ignored for vanilla YOLO. YAML list: `classes:="['bottle','hammer']"`. |
| `threshold` | `0.5` | Detection confidence floor. Drop to `0.01–0.1` for text-prompt mode, `0.005–0.05` for picture-prompt mode. |
| `imgsz_width` / `imgsz_height` | `960` / `544` | Inference resolution. Matches the D435i RGB stream. Try `640×384` for speed or `1280×736` for small-object recall. |
| `depth_image_units_divisor` | `1000` | RealSense depth is 16UC1 mm → divide by 1000 to get meters. Set to **1** in sim. |
| `target_frame` | (none) | Frame for the 3D detections. `camera_color_optical_frame` on real robot, `torso_link` in sim. |
| `use_3d` | `False` | Must be `True` to publish `/yolo/detections_3d`. |
| `use_tracking` | `True` | ByteTrack across frames. Disable for cleaner per-frame masks (and slightly faster). |
| `device` | `cuda:0` | Set to `cpu` if you don't have a GPU. CPU is fine at 11s/26s sizes and 960×544. |
| `maximum_detection_threshold` | `0.5` | Maximum allowed depth (m) for a detection's centroid. Bump to `5.0` for tabletop scenes. |
| `vp_enable` | `False` | Picture-prompt mode toggle. |
| `vp_refer_image` | `""` | Reference image path (from `save_one_frame.py`). |
| `vp_bboxes` | `[]` | YAML list of `[x1,y1,x2,y2]` boxes (from `vp_box_tool.py`). |
| `vp_cls` | `[]` | YAML list of class IDs matching `vp_bboxes`. |

**Tuning tips:**
- For vanilla YOLO and `best2.pt`, the trained class names are fixed — don't pass `classes:=`.
- For YOLOE text-prompt mode, picking the right prompt matters more than you'd think. "phone" works; "cell phone" works better. "small flashlight" works; "torch" is ambiguous (UK English / Olympic torch).
- Drop `threshold` first when you're not getting detections. If still nothing, check camera orientation and lighting.
- Bump `maximum_detection_threshold` if detections show up in 2D but the 3D point is missing — it gates how far behind the camera a centroid can be and still count.
- Picture-prompt mode is sensitive to the size of your reference box. A box that's too tight ignores context; too loose includes background. Aim for a tight box around just the object.

---

### `best_mask_pca.py`

Subscribes to YOLO detections + the depth image, picks the highest-confidence (or class-filtered) detection, extracts only the mask pixels, runs a connected-component "main mass" filter to drop noise, then computes the **centroid**, **principal axes (PCA)**, and **oriented bounding box (OBB)** of the resulting 3D point cloud. Publishes everything as RViz markers.

**Location:** `~/ros2_ws/tools/best_mask_pca.py`

**Run:**
```bash
python3 ~/ros2_ws/tools/best_mask_pca.py --ros-args \
  -p detections_topic:=/yolo/detections_3d \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info
```

**Publishes:**
- `/best_mask_pca_markers` — `MarkerArray`:
  - id 0: centroid (sphere)
  - ids 1, 2, 3: PCA axes (arrows, sorted by eigenvalue, longest first)
  - id 4: OBB edges (line list)
- `/best_mask_cloud` — `PointCloud2` of the filtered mask points

**Key parameters (set with `-p name:=value`):**

| Parameter | Default | What it does / when to change |
|---|---|---|
| `detections_topic` | `/yolo/detections_3d` | Where YOLO publishes its 3D detections. Match the namespace of the yolo launch. |
| `depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` | Aligned depth image. Use `/d435i/depth_image` in sim. |
| `camera_info_topic` | `/camera/camera/aligned_depth_to_color/camera_info` | Camera intrinsics. Must match `depth_topic`. |
| `output_frame` | (depth frame) | Frame for the published markers. Override to `torso_link` if you want markers in torso frame. |
| `projection_frame_override` | (empty) | Force a specific frame when reprojecting depth pixels — use `d435_optical_frame` in sim where the driver frame name differs from the URDF. |
| `depth_units_divisor` | `1000.0` | 16UC1 mm → m. Set to **1.0** in sim (depth is already meters). |
| `sample_stride` | `4` | Subsample factor. `1` = use every mask pixel (slower, denser); `4–6` is fine for grasp planning. |
| `mask_shrink_px` | `0` | Erode the mask by N pixels to drop edge noise. Try `1–3` if the OBB is being inflated by background. |
| `depth_band_m` | `0.20` | Z-band filter around the median mask depth. Keeps only points within ±band/2 m of the median. Narrow this (e.g. `0.10`) for thin objects, widen (`0.30`) for tall ones. |
| `use_main_component` | `True` | Keep only the largest connected component. Almost always leave on. |
| `min_component_pixels` | `20` | Drop components smaller than this. Raise for noisier data. |
| `min_score` | `0.0` | Confidence floor when picking the best detection. |
| `allowed_classes` | `[]` | If non-empty, restrict to these class names: `-p allowed_classes:='[cereal box,toothpaste]'`. |
| `max_points` | `40000` | Cap on PointCloud2 size (markers are unaffected). `0` = unlimited. |
| `publish_cloud` | `True` | Toggle the cloud topic (slight CPU saving if you don't view it in RViz). |
| `publish_markers` | `True` | Toggle the marker topic. |

**Tuning tips:**
- If the OBB axes look wrong on a flat object (toothpaste tube lying down), narrow `depth_band_m` to `0.08–0.12`. The default 0.20 includes the table surface, which inflates one axis.
- If you see PCA flipping orientation frame-to-frame for symmetric objects, that's normal — PCA axes are sign-ambiguous. `pca_obb_to_torso_pregrasp.py` handles this with `--direction-sign`.
- Drop `sample_stride` to `1–2` if your object is small or far away.

---

### `pca_obb_to_torso_pregrasp.py`

Reads the PCA/OBB markers, picks an approach axis, computes a right-hand-friendly pregrasp point with clearance offset, transforms it through `camera → torso_link` using Pinocchio FK on the URDF, and applies a hand-length offset so that the **hand center** ends up at the target (the IK script targets the wrist, which is ~10 cm proximal to the palm).

**Location:** `~/ros2_ws/tools/movement/pca_obb_to_torso_pregrasp.py`

**Run:**
```bash
python3 ~/ros2_ws/tools/movement/pca_obb_to_torso_pregrasp.py \
  --hand-offset 0.10 \
  --target-up-offset 0.025 \
  --approach-axis-id 2 \
  --clearance 0.04 \
  --debug
```

**Publishes:**
- `/right_pregrasp_camera_point` — `PointStamped` in camera optical frame (debug)
- `/right_pregrasp_torso_point` — `PointStamped` in `torso_link` (consumed by IK script)
- `/right_pregrasp_torso_markers` — `MarkerArray` for RViz visualization

**Key parameters:**

| Argument | Default | What it does / when to change |
|---|---|---|
| `--input-topic` | `/best_mask_pca_markers` | PCA markers from `best_mask_pca.py`. |
| `--approach-axis-id` | `2` | Which PCA axis to approach from: `1` = longest, `2` = middle, `3` = shortest. For a horizontal cereal box on a table, axis 2 (vertical face) is usually the side grip. |
| `--direction-sign` | `0` | Which way to approach along the axis. `0` = auto-pick the side with larger camera-x (right-hand approach), `+1` / `-1` to force. |
| `--clearance` | `0.04` (4 cm) | Standoff distance from the OBB face. Increase for taller / unstable objects. |
| `--max-standoff` | `0.20` (20 cm) | Cap on standoff (clearance + half OBB extent). Prevents the pregrasp ending up way out in space for very long objects. |
| `--hand-offset` | `0.10` (10 cm) | Distance from wrist (`right_wrist_yaw_link`) to hand center. Calibrated to the Dex3-1 with closed palm. **Don't change unless you're using a different end-effector.** |
| `--hand-offset-mode` | `"approach"` | `approach` = subtract along the approach vector (correct for top/side grasps), `none` = ignore (test mode). |
| `--target-forward-offset` | `0.00` | Bias the published torso-frame point in +X (forward). Use to nudge if grasps land slightly short / long of the object. |
| `--target-left-offset` | `0.00` | Bias in +Y (toward the robot's left). Positive = away from the right hand — useful if grasps over-shoot to the right. |
| `--target-up-offset` | `0.025` | Bias in +Z (up). Most-used tuning knob: raise to clear the table, lower to dive into thin objects. |
| `--ws-min-forward` / `--ws-max-forward` | `0.05` / `0.40` | Forward workspace. |
| `--ws-min-left` / `--ws-max-left` | `-0.40` / `0.00` | Left workspace (right arm only — max must stay ≤ 0). |
| `--ws-min-up` / `--ws-max-up` | `0.05` / `0.60` | Vertical workspace. |
| `--reject-outside-workspace` | off | If set, refuse to publish targets outside the workspace bounds. Otherwise publish + warn. |
| `--debug` | off | Verbose printing of each transform stage. Always use first. |

**Tuning tips:**
- Start with `--debug` on every new object. The script prints the chosen axis, the standoff, the camera-frame and torso-frame points, and the hand offset. Comparing these to what you see in RViz tells you which knob to turn.
- If grasps consistently miss by 1–2 cm in one direction, fix it with `--target-{forward,left,up}-offset` rather than tweaking `--clearance`. Clearance changes the approach geometry, the offsets just translate the final point.
- `--approach-axis-id` is the most common cause of "the grasp angle looks wrong." Try `1` vs `2` vs `3` and watch which orientation the arrow markers take.

---

### `g1_ik_ready_then_target_dex3_table.py`

The core motion script. Real-robot execution path:

1. Solve Pinocchio IK for `right_wrist_yaw_link` at the target in torso frame.
2. Connect SDK2 (`rt/lowstate`, `rt/arm_sdk`, optional `rt/dex3/right/cmd`).
3. Read current arm state, pin left arm.
4. Ramp `arm_sdk` weight 0 → 1.
5. Swing right shoulder roll outward for clearance.
6. Optional **desk-clearance** waypoint: shoulder back, elbow up (avoids tabletop on the way in).
7. Move right arm to READY.
8. Optional Dex3 OPEN at READY.
9. Move READY → IK target.
10. Optional Dex3 CLOSE with pressure-gated early stop.
11. Optional lift / lower / release sequence.
12. Optional desk-exit waypoint, then ramp weight 1 → 0.

**Location:** `~/ros2_ws/tools/movement/g1_ik_ready_then_target_dex3_table.py`

**Two ways to run.**

**(a) Fixed-point grasp** — supply the target manually:

```bash
python3 ~/ros2_ws/tools/movement/g1_ik_ready_then_target_dex3_table.py \
  --target-forward 0.12 --target-left -0.18 --target-up 0.00 \
  --use-dex3 --hand-open-at-ready --hand-close-at-target \
  --hand-closed-q "0.0120,0.1086,-1.3349,1.1001,1.1479,1.2953,0.8418" \
  --hand-pressure-stop --hand-pressure-delta-threshold 1000 \
  --use-desk-clearance \
  --log-errors
```

**(b) Subscribe-mode** — take the target from `pca_obb_to_torso_pregrasp.py`:

```bash
python3 ~/ros2_ws/tools/movement/g1_ik_ready_then_target_dex3_table.py \
  --subscribe-target --target-topic /right_pregrasp_torso_point \
  --subscribe-timeout 30.0 --subscribe-settling 1.0 \
  --use-desk-clearance \
  --use-dex3 --hand-open-at-ready --hand-close-at-target \
  --hand-closed-q "0.0120,0.1086,-1.3349,1.1001,1.1479,1.2953,0.8418" \
  --hand-pressure-stop --hand-pressure-delta-threshold 1000 \
  --log-errors
```

**Key parameters — target & workspace:**

| Argument | Default | What it does |
|---|---|---|
| `--target-forward` / `--target-left` / `--target-up` | – | IK target in `torso_link`. Required unless `--subscribe-target`. |
| `--subscribe-target` | off | Wait for a `PointStamped` on `--target-topic`. |
| `--target-topic` | `/right_pregrasp_torso_point` | Where subscribe mode listens. |
| `--subscribe-timeout` | `30.0` s | Give up if no target arrives in this window. |
| `--subscribe-settling` | `1.0` s | After receiving, wait this long before acting (lets averaging stabilize). |
| `--ws-{min,max}-{forward,left,up}` | as in `pca_obb_to_torso_pregrasp` | Hard workspace bounds. Targets outside are rejected. |

**Key parameters — motion & gains:**

| Argument | Default | What it does |
|---|---|---|
| `--kp-test` | `35` | Default position gain on all arm joints. Raise for stiffer tracking; lower if motion looks aggressive. |
| `--right-shoulder-roll-kp` | `45` | Dedicated `kp` for shoulder roll during clearance stage. |
| `--right-elbow-kp` | `65` | Dedicated `kp` for elbow (needs to hold against gravity). |
| `--kp-hold` | `40` | `kp` during hold-at-target. |
| `--kd` | `1.5` | Velocity gain. Rarely changed. |
| `--ik-gradual-step` | `1.0` | Fraction of the IK delta to apply. **Always start at `0.25` for a new pose**, then `0.5`, then `1.0`. |
| `--weight-ramp` | `1.5` s | Time to ramp arm_sdk weight 0→1. |
| `--clearance-duration` | `3.0` s | Time for the shoulder-roll-out waypoint. |
| `--ready-duration` | `8.0` s | Time from current pose to READY. |
| `--target-duration` | `10.0–12.0` s | Time from READY to IK target. Lengthen for larger Cartesian moves. |
| `--target-hold` | `8.0` s | Hold time at the target (during which Dex3 closes). |
| `--release` | `2.0` s | Time to ramp arm_sdk weight 1→0 at the end. |
| `--force-safe-shoulder-roll` | off | Pin shoulder roll to `--safe-shoulder-roll` during ready/target. |
| `--safe-shoulder-roll` | `-0.30` rad | Outward bias for right shoulder roll. More negative = further out. |
| `--dry-run` | off | Compute IK and print, but don't move the robot. |
| `--ready-only` | off | Move to READY and stop. |
| `--log-errors` | off | Verbose per-joint tracking-error logging. Always use during tuning. |
| `--iface` | `enx00e04c7015d3` | DDS network interface. |

**Key parameters — desk clearance:**

The desk-clearance pose was recorded by physically posing the robot in damping mode and reading off joint values. Use the values in the example commands as-is unless your tabletop height is very different.

| Argument | Default | Description |
|---|---|---|
| `--use-desk-clearance` | off | Enable the table-avoidance waypoint. |
| `--desk-clearance-shoulder-pitch` | `1.1497` | Shoulder pitch at the clearance pose (back & up). |
| `--desk-clearance-shoulder-roll` | `-0.6680` | Shoulder roll (out from body). |
| `--desk-clearance-shoulder-yaw` | `0.0836` | Shoulder yaw. |
| `--desk-clearance-elbow` | `-0.3022` | Elbow bend up. **Negative** because positive lowers the hand on this robot. |
| `--desk-clearance-wrist-{roll,pitch,yaw}` | small values | Wrist neutral at clearance. |
| `--desk-shoulder-duration` / `--desk-elbow-duration` | `8.0` s each | Stage timing into the clearance pose. |
| `--desk-exit-shoulder-duration` / `--desk-exit-elbow-duration` | `8.0` s each | Stage timing out of the clearance pose. |

**Key parameters — Dex3 hand:**

| Argument | Default | What it does |
|---|---|---|
| `--use-dex3` | off | Enable hand control. Auto-enabled if any other hand flag is set. |
| `--hand-cmd-topic` | `rt/dex3/right/cmd` | DDS command topic. |
| `--hand-state-topic` | `auto` | Subscribe to both `rt/dex3/right/state` and `rt/lf/dex3/right/state`. |
| `--hand-open-q` | preset 7-float string | Open pose joints (thumb_flex_0 ≈ +0.577, fingers near 0). |
| `--hand-closed-q` | **none** | Closed pose, comma-separated 7 floats. **Required** if `--hand-close-at-target`. No default — must be calibrated per-object. |
| `--hand-open-at-ready` | off | Open hand at READY before moving to target. |
| `--hand-close-at-target` | off | Close hand at IK target. |
| `--hand-open-duration` | `2.0` s | Time to open. |
| `--hand-close-duration` | `4.0` s | Time to close (interrupted by pressure stop). |
| `--hand-hold-after-close` | `2.0` s | Hold the squeezed pose this long. |
| `--hand-kp` / `--hand-kd` | `0.8` / `0.1` | Finger gains. |
| `--hand-pressure-stop` | off | Stop the close early when pressure delta exceeds threshold. |
| `--hand-pressure-delta-threshold` | `3000.0` | Pressure delta over baseline to trigger stop. **Tune per object**: ~150 for very light/thin (use `--hand-pressure-stop-mode mean`), 1000 for typical objects, 3000 for rigid heavy. |
| `--hand-pressure-stop-mode` | `peak` | `peak` = trigger on any channel; `mean` = trigger on average across channels (better for distributed contact). |
| `--hand-close-multiplier` | `1.0` | Scale the closed pose toward open (>1.0) or past closed (<1.0). Use ~`1.1` for "barely-closed" grips on thin objects. |

**Recording a closed pose for a new object:**

1. Put the robot in damping mode (low gains, free movement).
2. Place the object in the hand.
3. Manually close the fingers around it.
4. In another terminal: `python3 ~/ros2_ws/tools/movement/right_arm_joint_probe.py` (or the equivalent hand-state echo) and read off the 7 joint values for the right Dex3.
5. Paste them as `--hand-closed-q "j0,j1,j2,j3,j4,j5,j6"`.

**Tuning tips:**
- **Always start with `--dry-run` + `--log-errors`** for a new target. Confirm the IK solution is sensible before letting it run.
- For the very first run after any pose change, set `--ik-gradual-step 0.25` and ramp up.
- If the arm slams into READY or the target, lengthen the corresponding `--*-duration` flag. 10–12 s is normal for large Cartesian moves; don't go below 6 s on the real robot.
- If you're getting workspace rejection on a target you trust, override the relevant `--ws-*` flag — don't disable bounds globally.

---

## End-to-end demo

Five terminals (each needs `sros` first). This is the working pipeline for a Hugging Face `best2.pt` grasp of a cereal box / torch / toothpaste, or any YOLOE class on the table.

**Terminal 1 — RealSense:**
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true
```

**Terminal 2 — Detection** (pick one):

```bash
# Option A: custom model (cereal box / torch / toothpaste)
ros2 launch yolo_bringup yolo.launch.py \
  model:=/home/$USER/ros2_ws/Models/best2.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 imgsz_width:=960 imgsz_height:=544 \
  depth_image_units_divisor:=1000 maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame use_3d:=True device:=cpu

# Option B: vanilla YOLO11 (COCO classes — bottle, cup, mouse, keyboard, scissors, ...)
ros2 launch yolo_bringup yolo.launch.py \
  model:=/home/$USER/ros2_ws/Models/yolo11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 imgsz_width:=960 imgsz_height:=544 \
  depth_image_units_divisor:=1000 maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame use_3d:=True device:=cpu

# Option C: YOLOE text-prompt (open vocab, any class)
ros2 launch yolo_bringup yolo.launch.py \
  model_type:=YOLOE \
  model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  input_depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  input_depth_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  yolo_encoding:=rgb8 imgsz_width:=960 imgsz_height:=544 \
  threshold:=0.10 \
  classes:="['mug','bottle','keyboard','mouse','phone','hammer']" \
  depth_image_units_divisor:=1000 maximum_detection_threshold:=5.0 \
  target_frame:=camera_color_optical_frame use_3d:=True device:=cpu

# Option D: YOLOE picture-prompt (visual reference)
# First: python3 ~/ros2_ws/tools/save_one_frame.py
# Then:  python3 ~/ros2_ws/tools/vp_box_tool.py ~/Pictures/ref_from_cam.png
# Paste the printed vp_bboxes / vp_cls into the launch below:
ros2 launch yolo_bringup yoloe.launch.py \
  model:=/home/$USER/ros2_ws/Models/yoloe-11s-seg.pt \
  input_image_topic:=/camera/camera/color/image_raw \
  device:=cpu use_3d:=False \
  vp_enable:=True \
  vp_refer_image:=/home/$USER/Pictures/ref_from_cam.png \
  vp_bboxes:='[[221,405,345,858]]' \
  vp_cls:='[0]' \
  threshold:=0.005
```

**Terminal 3 — PCA / OBB:**
```bash
python3 ~/ros2_ws/tools/best_mask_pca.py --ros-args \
  -p detections_topic:=/yolo/detections_3d \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info
```

**Terminal 4 — Pregrasp planner:**
```bash
python3 ~/ros2_ws/tools/movement/pca_obb_to_torso_pregrasp.py \
  --hand-offset 0.10 --target-up-offset 0.025 \
  --approach-axis-id 2 --clearance 0.04 --debug
```

**Terminal 5 — IK + arm + Dex3:**
```bash
python3 ~/ros2_ws/tools/movement/g1_ik_ready_then_target_dex3_table.py \
  --subscribe-target --target-topic /right_pregrasp_torso_point \
  --subscribe-timeout 30.0 --subscribe-settling 1.0 \
  --use-desk-clearance \
  --use-dex3 --hand-open-at-ready --hand-close-at-target \
  --hand-closed-q "0.0120,0.1086,-1.3349,1.1001,1.1479,1.2953,0.8418" \
  --hand-pressure-stop --hand-pressure-delta-threshold 1000 \
  --log-errors
```

Keep L2+B (emergency stop) accessible at all times.

---

## Simulation (optional)

For a Gazebo-only test path:

```bash
# Terminal 1 — Ignition
export IGN_GAZEBO_RESOURCE_PATH=$HOME/gazebo_g1_ws/src:$IGN_GAZEBO_RESOURCE_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/gazebo_g1_ws/src:$GZ_SIM_RESOURCE_PATH
ign gazebo ~/gazebo_g1_ws/worlds/g1_harness_world.sdf

# Terminal 2 — robot_state_publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args \
  -p robot_description:="$(cat ~/gazebo_g1_ws/src/g1_description/urdf/g1_29dof_d435i.urdf)"

# Terminal 3 — controllers
ros2 run controller_manager spawner joint_state_broadcaster --switch-timeout 30
ros2 run controller_manager spawner whole_body_controller --switch-timeout 30

# Terminal 4 — camera bridge
ros2 run ros_gz_bridge parameter_bridge \
  /d435i/image@sensor_msgs/msg/Image[ignition.msgs.Image \
  /d435i/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image \
  /d435i/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo

# Then YOLO + best_mask_pca with depth_image_units_divisor:=1 and projection_frame_override:=d435_optical_frame
```

In sim, the IK script's SDK2 path doesn't apply — drive joints via `/whole_body_controller/joint_trajectory` instead.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ros2 topic list` empty against the real robot | `unitree_ros2/setup.sh` was sourced **after** the daemon started. Re-source in correct order, then `ros2 daemon stop && ros2 daemon start`. |
| `cv_bridge` crash with NumPy errors | Ultralytics pulled NumPy 2.x. Re-pin: `pip install "numpy==1.26.4" "opencv-python==4.10.0.84"`. |
| `PytorchStreamReader failed reading zip archive` | The `.pt` file is corrupt. Check `ls -lh ~/ros2_ws/Models/` — `best2.pt` should be **6.63 MB**. If smaller, redownload. |
| YOLOE text-prompt model errors with MobileCLIP | `mobileclip2_b.ts` is corrupt or missing in the Ultralytics install. Workaround: use `yoloe-11s-seg-pf.pt` (prompt-free) or switch to YOLOE picture-prompt mode. |
| Best-mask PCA markers in wrong frame in sim | Set `-p projection_frame_override:=d435_optical_frame` and `-p depth_units_divisor:=1.0`. |
| IK rejects a target you trust | Override the offending `--ws-*` flag explicitly. Don't disable bounds entirely. |
| Robot collides with right thigh on the way to READY | `--force-safe-shoulder-roll --safe-shoulder-roll -0.30` (or more negative). |
| Robot collides with tabletop on the way to a low target | `--use-desk-clearance` with the recorded waypoint. |

---

## License

`yolo_node.py` and the `yolo*launch*.py` files are derived from [`yolo_ros`](https://github.com/mgonzs13/yolo_ros) by Miguel Ángel González Santamarta and remain under **GPL-3.0**. 

The custom `best2.pt` model is hosted at [`calebandchill/cerealbox.torch.toothpaste`](https://huggingface.co/calebandchill/cerealbox.torch.toothpaste).
