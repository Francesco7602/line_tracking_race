# Line Tracking Race ROS2 Jazzy + Gazebo Harmonic Version (Work in progress)
This repository contains a ROS2 project for a simulated line-following race car. The primary goal is to navigate a track defined by yellow lines as quickly and accurately as possible. The project is built for ROS2 Jazzy and uses Gazebo Harmonic for simulation. It serves as a comprehensive platform for developing and testing advanced planning and control strategies for autonomous racing.

## 1. Installation

### 1.1. Prerequisites
Ensure you have a working ROS2 Jazzy installation and the necessary tools.
```bash
sudo apt update && sudo apt install python3-vcstool python3-colcon-common-extensions git wget
```

### 1.2. Clone and Build
Clone the repository into your ROS2 workspace (e.g., `~/ros2_ws/src`) and build the packages.

```bash
# Clone the repository
git clone <your-repository-git-url>

# Navigate to your workspace root
cd ~/ros2_ws

# Install dependencies
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build
```

## 2. Usage

After building, source the workspace and launch the simulation.

### 2.1. Source the Workspace
```bash
source ~/ros2_ws/install/setup.bash
```

### 2.2. Launch the Application
This project features multiple planning strategies. The most advanced is the `exploration_based` strategy, which can be run with the following command:

```bash
ros2 launch line_tracking_race_application full_launch.py strategy:=exploration_based
```
This command launches the Gazebo simulation, the robot model, and all the necessary nodes for the car to start navigating the track.

## 3. System Architecture

The project follows a modular architecture separating perception, planning, and control. The core logic resides in the `line_tracking_race_application` package.

### 3.1. Control Node: `better_control_node`

This is an advanced PID-based controller with several enhancements for high-performance racing:

*   **Dynamic Gain Scheduling**: The controller subscribes to the `/planner/mode` topic and dynamically adjusts its PID gains. It uses a more aggressive set of gains during the `Exploitation` phase (mode `1.0`) for faster driving and a safer set during the `Exploration` phase (mode `0.0`).
*   **Feed-Forward Control**: It uses track curvature information, published on the `/planning/curvature` topic, to proactively apply steering adjustments. This allows the car to anticipate turns rather than just reacting to errors.
*   **Curvature-Based Velocity Scaling**: The controller adjusts the robot's target velocity based on the current track curvature, automatically slowing down for sharp turns and accelerating on straight sections.
*   **Advanced Saturation Logic**: It implements dynamic saturation for both linear and angular velocity. This prevents control instability at high speeds and ensures the robot remains stable by reducing speed during aggressive turns.

### 3.2. Planning Strategies

The planner node can be configured with different strategies. The most notable ones are `better_centerline_strategy` and `exploration_based_strategy`.

#### `better_centerline_strategy`

This is a robust perception and local planning strategy. Its main function is to process camera images to identify the track and compute a steering error.

1.  **Image Processing**: It converts the raw camera image to the HSV color space to robustly detect the yellow track lines.
2.  **Boundary Extraction**: It identifies the left and right boundaries of the track from the masked image.
3.  **Centerline Computation**: A robust centerline is calculated by interpolating between the detected left and right boundaries. This provides a stable path even if one of the boundaries is partially obscured.
4.  **Error Calculation**: It selects a waypoint on the centerline ahead of the vehicle and calculates the angle error between the robot's current heading and the direction to the waypoint. This error is published for the control node.
5.  **Positional Error**: It also calculates the current lateral (positional) error relative to the centerline and publishes it for logging and analysis.

#### `exploration_based_strategy`

This is a high-level strategy that orchestrates the entire racing mission. It operates in two distinct phases: an initial **Exploration** phase to learn the track layout, followed by an **Exploitation** phase to race at maximum speed.

**Phase 1: Exploration**

*   **Goal**: To drive cautiously and build a complete, accurate map of the track.
*   **Operation**:
    1.  **Local Navigation**: It relies on the `better_centerline_strategy` to follow the track lines detected by the camera.
    2.  **Map Building**: As the robot moves, it uses a perspective transform to convert the detected track lines from camera image coordinates to world coordinates. This data is combined with the robot's odometry to populate a 2D occupancy grid map representing the entire circuit.
    3.  **Loop Closure Detection**: The strategy continuously monitors the robot's position. When it detects that the robot has returned to a previously mapped area after traveling a significant distance, it determines that the track loop is complete and transitions to the next phase.
*   **Mode Publication**: During this phase, it publishes `0.0` to the `/planner/mode` topic to signal that the robot is in exploration mode.

**Phase 2: Exploitation**

*   **Goal**: To use the completed map to drive as fast as possible by optimizing the racing line and speed.
*   **Operation**:
    1.  **Trajectory Optimization**: Upon entering exploitation mode, the strategy processes the generated map. It applies skeletonization to extract a precise, one-pixel-thin centerline representing the optimal raceline.
    2.  **Velocity Profile Generation**: It calculates the curvature at every point along this new raceline. Based on a maximum lateral acceleration limit, it computes a velocity profile that assigns the highest possible speed for each segment of the track—slowing down for sharp turns and accelerating on straights.
    3.  **High-Speed Execution**: During the race, the robot follows the optimized trajectory. It uses a moving window search to efficiently find its position on the pre-computed path and retrieves the target velocity from the profile. It also looks ahead on the map to predict upcoming curvature.
    4.  **Control Feed-Forward**: The target velocity and upcoming track curvature are published to the `better_control_node`, allowing it to use aggressive gains and proactive steering adjustments.
*   **Mode Publication**: During this phase, it publishes `1.0` to the `/planner/mode` topic, signaling the controller to switch to its high-performance settings.
