from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():

    # Setup project paths
    pkg_project_bringup = get_package_share_directory("line_tracking_race_bringup")
    pkg_project_gazebo = get_package_share_directory("line_tracking_race_gazebo")
    pkg_project_description = get_package_share_directory("line_tracking_race_description")
    
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")

    # Declare launch arguments

    world_file = DeclareLaunchArgument(
        'world_file',
        default_value='race_track.world',
        description='The world file'
    )

    car_file = DeclareLaunchArgument(
        'car_file',
        default_value='car.urdf.xacro',
        description='The car model file (xacro/urdf/sdf)'
    )

    x_pos_arg = DeclareLaunchArgument(
        "x_pos", default_value="-3.35", description="X position to spawn the robot" #per mappa base
        #"x_pos", default_value="-5.5", description="X position to spawn the robot"  # per mappa esagono
        #"x_pos", default_value="-1.9", description="X position to spawn the robot" #per mappa a otto
    )
    y_pos_arg = DeclareLaunchArgument(
        "y_pos", default_value="0.5", description="Y position to spawn the robot"#per mappa base
        #"y_pos", default_value="0.0", description="Y position to spawn the robot"  # per mappa esagono
        #"y_pos", default_value="1.7", description="Y position to spawn the robot"#per mappa a otto
    )
    z_pos_arg = DeclareLaunchArgument(
        "z_pos", default_value="0.45", description="Z position to spawn the robot"#per mappa base
        #"z_pos", default_value="0.45", description="Z position to spawn the robot"  # per mappa esagono
        #"z_pos", default_value="0.45", description="Z position to spawn the robot"#per mappa a otto
    )
    yaw_arg = DeclareLaunchArgument(
        "yaw", default_value="1.54", description="Yaw orientation to spawn the robot"#per mappa base
        #"yaw", default_value="1.54", description="Yaw orientation to spawn the robot"  # per mappa esagono
        #"yaw", default_value="7.0", description="Yaw orientation to spawn the robot"#per mappa a otto
    )


    # launch gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={
            'gz_args': [PathJoinSubstitution([pkg_project_gazebo, 'worlds', LaunchConfiguration("world_file")]), " -v"],
            'on_exit_shutdown': 'True'
        }.items(),
    )
    # Launch bridge separately, avoiding ros_gz_bridge.launch,
    # using the executable ros_gz_bridge/parameter_bridge
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': PathJoinSubstitution([pkg_project_bringup, "config", "ros_gz_bridge.yaml"]),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    # --------- Load car xacro/urdf ---------
    # Start a robot_state_publisher node publishing a "robot_description" topic
    models_path = PathJoinSubstitution([pkg_project_description, "models"])

    car_urdf = IncludeLaunchDescription(
        PathJoinSubstitution(
            [
                get_package_share_directory("urdf_launch"),
                "launch",
                "description.launch.py",
            ]
        ),
        launch_arguments={
            "urdf_package": "line_tracking_race_description",
            "urdf_package_path": PathJoinSubstitution(
                [models_path, "urdf", LaunchConfiguration("car_file")]
            ),
        }.items(),
    )

    # --------- Spawn Robot node ---------
    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "car",
            "-topic", "robot_description",
            "-x", LaunchConfiguration("x_pos"),
            "-y", LaunchConfiguration("y_pos"),
            "-z", LaunchConfiguration("z_pos"),
            "-Y", LaunchConfiguration("yaw"),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            world_file,
            car_file,
            x_pos_arg,
            y_pos_arg,
            z_pos_arg,
            yaw_arg,
            gz_sim,
            gz_bridge,
            car_urdf,
            spawn_robot_node,
        ]
    )
