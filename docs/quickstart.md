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

Mission simulation with obstacle lane-change avoidance:

```bash
qqq
ros2 launch simulation_pkg auto_drive.launch.py
```

In `auto_drive.launch.py`, traffic lights are still handled by `motion_planner_node`,
but lidar obstacles are routed to `navigator_node` so the vehicle changes lanes
instead of stopping in front of an obstacle vehicle. To test the old stop-at-obstacle
behavior, run:

```bash
qqq
ros2 launch simulation_pkg auto_drive.launch.py use_avoidance:=false motion_lidar_topic:=lidar_obstacle_info
```
