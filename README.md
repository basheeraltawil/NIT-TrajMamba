[![build](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/build.yaml/badge.svg)](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/build.yaml)[![jazzy-ci](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/jazzy-ci.yaml/badge.svg)](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/jazzy-ci.yaml)[![test](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/test.yaml/badge.svg)](https://github.com/ovgu-nit/nit_human_traj_estimation/actions/workflows/test.yaml)
---
# nit_human_traj_estimation
ROS2 package for multi-person 3D trajectory estimation and future-trajectory prediction, using MediaPipe Pose (hip-centre detection + depth deprojection) and a TrajMamba (Mamba SSM) model for prediction.

![Human Trajectory Estimation](docs/human_traj.png)

## Nav2 social costmap integration

This package's predicted-trajectory `MarkerArray` output feeds a Nav2 local
costmap layer via two sibling packages in `human_traj_nav/`:
`social_traj_bridge` and `social_traj_costmap_plugin`. See
[`human-traj-social-nav-integration.md`](human-traj-social-nav-integration.md)
for the full design, and [`../README.md`](../README.md) for the three-package
layout.

## Running the Node

### Option A — via nit_smach (robot deployment)
```bash
ros2 launch nit_smach run_module.launch.py robot:=tiago module:=nit_human_traj_estimation module_pkg:=nit_human_traj_estimation module_launch_file:=human_traj_estimation_launch.py
ros2 run nit_smach activate_module_sm --ros-args -p module:=nit_human_traj_estimation -p activate:=true -p robot:=tiago
ros2 run nit_smach activate_module_sm --ros-args -p module:=nit_human_traj_estimation -p activate:=false -p robot:=tiago
```

### Option B — direct launch
```bash
ros2 launch nit_human_traj_estimation human_traj_estimation_launch.py
```

If you're running from a conda environment with a Jetson-matched `torch` build for GPU (see [GPU note](#gpu-on-jetson) below):
```bash
conda activate trajectory
export PYTHONPATH=/home/basheer/miniforge3/envs/trajectory/lib/python3.10/site-packages:/home/basheer/.local/lib/python3.10/site-packages:/usr/lib/python3/dist-packages
export LD_LIBRARY_PATH=/home/basheer/miniforge3/envs/trajectory/lib
source ~/basheer_ws/install/setup.bash
ros2 launch nit_human_traj_estimation human_traj_estimation_launch.py
```

#### Activate / deactivate processing
```bash
ros2 service call /nit_human_traj_estimation/activate std_srvs/srv/SetBool "{data: true}"
ros2 service call /nit_human_traj_estimation/status std_srvs/srv/Trigger "{}"
```

#### View the result
- Local live window (default `enable_window:=true`): skeletons, observed trail, predicted trajectory with uncertainty cone, depth gauge, HUD. Keys: `q` quit, `s` screenshot, `h` toggle HUD, `k` toggle skeleton.
- Or subscribe to the output topics below (works regardless of `enable_window`).

## Requirements
- ROS 2 Humble
- A trained TrajMamba checkpoint (`best_model.pth`) and a MediaPipe `pose_landmarker.task` file. Auto-downloaded on first run from `checkpoints_url` if not already present locally (see [`checkpoints/README.md`](checkpoints/README.md)) — requires the `requests` pip package and network access; falls back to whatever's already local if the download fails.
- Python deps not covered by `rosdep`: see `requirements.txt` (`torch`, `mediapipe`, `requests`)
- Synchronized RGB + depth + CameraInfo topics (e.g. RealSense, TIAGo head camera)

### GPU on Jetson
Generic `pip install torch` resolves to a build compiled against a newer CUDA version than the Jetson driver stack supports, and silently falls back to CPU. Use NVIDIA's Jetson-specific `torch` wheel (matching your JetPack/L4T version — check `cat /etc/nv_tegra_release`) instead. This package uses the `trajectory` conda env, which already has a working Jetson `torch` build. Two things matter for it to actually take effect:

1. **`export PYTHONPATH`** so the conda env can see ROS's own system packages (`catkin_pkg`, etc.) alongside its own — without this, `rqt`-family tools and some ROS introspection break inside the conda env:
```bash
   conda activate trajectory
   export PYTHONPATH=/home/basheer/miniforge3/envs/trajectory/lib/python3.10/site-packages:$PYTHONPATH:/usr/lib/python3/dist-packages
```
2. **Build with the same env active that you'll launch from** — the built executable's shebang line is pinned to whichever `python3` was active at `colcon build` time, and does *not* change just because you `conda activate` something else afterward:
```bash
   conda activate trajectory
   cd ~/basheer_ws
   rm -rf build/nit_human_traj_estimation install/nit_human_traj_estimation log
   colcon build --packages-select nit_human_traj_estimation
```

Check `Device: cuda` (not `cpu`) in the node's startup log to confirm.

## Parameters
All parameters live in [`config/config.yaml`](config/config.yaml) and are loaded by the launch file — edit that file for a deployment, or pass `config_file:=/path/to/other.yaml` to use a different one.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | string | `checkpoints/best_model.pth` | TrajMamba checkpoint (relative → resolved against install share dir) |
| `pose_landmarker_task` | string | `checkpoints/pose_landmarker.task` | MediaPipe Pose Landmarker task file |
| `checkpoints_url` | string | *(OVGU cloud share)* | Nextcloud folder share to auto-download missing checkpoint file(s) from — see [`checkpoints/README.md`](checkpoints/README.md) |
| `rgb_topic` | string | `/head_front_camera/rgb/image_raw` | Input RGB image topic (`Image` or `.../compressed`) |
| `depth_topic` | string | `/head_front_camera/depth/image_raw` | Input depth image topic |
| `camera_info_topic` | string | `/head_front_camera/rgb/camera_info` | Input camera info (required before any frame is processed) |
| `frame_id` | string | `head_front_camera_rgb_frame` | Frame ID for published messages |
| `output_image_topic` | string | `nit_human_traj_estimation/image` | Output: annotated visualization image |
| `marker_topic` | string | `nit_human_traj_estimation/markers` | Output: predicted trajectories as `MarkerArray` (line strips) |
| `pose_array_topic` | string | `nit_human_traj_estimation/predicted_trajectories` | Output: predicted trajectories as `PoseArray` |
| `activation` | bool | `false` | Whether the node is processing frames at startup (matches `nit_pose_estimation`'s convention — start idle, let `nit_smach` activate) |
| `activation_service_name` | string | `/nit_human_traj_estimation/activate` | `SetBool` service to activate/deactivate processing |
| `status_service_name` | string | `/nit_human_traj_estimation/status` | `Trigger` service reporting active/inactive |
| `enable_window` | bool | `true` | Show local `cv2` live window (set `false` for headless robot deployment — outputs still publish either way) |
| `window_name` | string | `TrajMamba ROS Live` | Title of the cv2 window |
| `show_hud` | bool | `true` | Show FPS/status HUD overlay |
| `show_skeleton` | bool | `true` | Draw MediaPipe pose skeleton overlay |
| `screenshot_dir` | string | `figures` | Directory for `s`-key screenshots (relative → resolved against install share dir) |
| `queue_size` | int | `10` | RGB/depth `ApproximateTimeSynchronizer` queue size |
| `sync_slop` | double | `0.08` | RGB/depth synchronization tolerance (seconds) |
| `max_people` | int | `4` | Max simultaneously tracked people |
| `max_match_dist` | double | `0.8` | Max 3D distance (m) to associate a detection with an existing track |
| `max_misses` | int | `15` | Frames a track survives with no matching detection before being dropped |

### Prediction behavior
Each tracked person needs `obs_len` observed frames (from the checkpoint's config, typically 20) before TrajMamba starts predicting; until then they're drawn as an observed-only trail. Multi-person tracking is a simple nearest-neighbor associator (`max_match_dist`/`max_misses`), independent per person — TrajMamba runs once per tracked person, per frame.

## Training / Evaluation
`training/` contains the TrajMamba training pipeline — **not** part of the installed ROS package (no `__init__.py`, not picked up by `setup.py`'s `find_packages()`), since it needs `matplotlib`/full `torch` and has nothing to do with the ROS runtime.

```bash
cd training
python3 train_mamba.py --dataset_roots /path/to/rgbd_bonn_dataset --checkpoint_dir ./checkpoints
python3 evaluate.py --checkpoint ./checkpoints/best_model.pth --dataset_roots /path/to/rgbd_bonn_dataset --plot
```

Note: `training/mamba_model.py` is intentionally a separate copy from `nit_human_traj_estimation/mamba_model.py` (the runtime one) — the training pipeline stays a standalone, non-ROS tool, while the deployed node bundles its own copy so it doesn't depend on anything outside the installed package. Keep both in sync if you change the architecture.

## CI
- `build.yaml` — builds the package with `colcon` on ROS 2 Humble on every push.
- `jazzy-ci.yaml` — same, but against ROS 2 Jazzy.
- `test.yaml` — runs `ament` lint/style checks (non-blocking).
