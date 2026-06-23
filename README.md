# nav-vla

ROS 2 Jazzy workspace for autonomous-driving simulation and vision-language navigation experiments.

This repository is currently a skeleton for the main project. It keeps the reusable ROS 2 packages, Gazebo assets, launch files, and message definitions needed to start development without carrying experiment-specific datasets, generated results, or model weights.

## Packages

| Path | Purpose |
| --- | --- |
| `src/interfaces_pkg` | Shared ROS 2 message definitions |
| `src/simulation_pkg` | Gazebo worlds, vehicle models, launch files, and simulation helpers |
| `src/camera_perception_pkg` | Camera perception and YOLO integration nodes |
| `src/lidar_perception_pkg` | LiDAR publishing, processing, and obstacle detection nodes |
| `src/decision_making_pkg` | Path planning, motion planning, and driving decision nodes |
| `src/debug_pkg` | Logging, visualization, and debug utilities |

## Repository Policy

The repository stores source code, launch/config files, ROS interface definitions, and lightweight assets needed for development.

The following are intentionally kept out of Git:

- colcon outputs: `build/`, `install/`, `log/`
- Python caches and local virtual environments
- model weights such as `*.pt`, `*.onnx`, and `*.engine`
- ROS bags, databases, generated evaluation outputs, and large sample datasets

Model files should be copied into the expected local paths after cloning.

## Setup

```bash
cd ~/ROS2_project/nav-vla
sh install.sh
source ~/.bashrc
```

For a clean ROS dependency pass:

```bash
export AMENT_PREFIX_PATH=''
export CMAKE_PREFIX_PATH=''
source /opt/ros/jazzy/setup.bash
PYTHONNOUSERSITE=1 rosdep install -i --from-path src --rosdistro jazzy -y
```

## Build

```bash
cd ~/ROS2_project/nav-vla
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/local_setup.bash
```

## Run

```bash
cd ~/ROS2_project/nav-vla
source /opt/ros/jazzy/setup.bash
source install/local_setup.bash
export ROS_DOMAIN_ID=32
```

Driving simulation:

```bash
qqq
ros2 launch simulation_pkg driving_sim.launch.py
```

Mission simulation:

```bash
qqq
ros2 launch simulation_pkg mission_sim.launch.py
```

See [docs/quickstart.md](docs/quickstart.md) for a compact command list.
