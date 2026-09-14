#!/usr/bin/env python3
"""
Launch file for NIT Human Trajectory Estimation Node
Tracks people in 3D from RGB-D images and predicts their future trajectories
using TrajMamba.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('nit_human_traj_estimation'),
            'config',
            'config.yaml'
        ]),
        description='Path to the config file'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the human trajectory estimation node'
    )

    # Node
    human_traj_estimation_node = Node(
        package='nit_human_traj_estimation',
        executable='human_traj_estimation',  # This should match the entry point name in setup.py
        name='nit_human_traj_estimation',    # This can be any name you want for the node
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        config_file_arg,
        namespace_arg,
        human_traj_estimation_node
    ])
