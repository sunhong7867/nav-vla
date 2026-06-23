# Quickstart

## 1. Install

```bash
cd ~/ROS2_project/nav-vla
sh install.sh
source ~/.bashrc
```

Install ROS dependencies:

```bash
export AMENT_PREFIX_PATH=''
export CMAKE_PREFIX_PATH=''
source /opt/ros/jazzy/setup.bash
PYTHONNOUSERSITE=1 rosdep install -i --from-path src --rosdistro jazzy -y
```

## 2. Build

```bash
cd ~/ROS2_project/nav-vla
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/local_setup.bash
```

## 3. Run

Run these in each terminal before launching:

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
