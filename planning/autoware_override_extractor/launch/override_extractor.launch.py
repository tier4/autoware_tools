#!/usr/bin/env python3

# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_override_extractor")
    default_config = os.path.join(pkg_dir, "config", "override_extractor.param.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "input_dir",
                default_value="",
                description="Input directory containing rosbags",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="",
                description="Output directory for extracted segments",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to parameter file",
            ),
            Node(
                package="autoware_override_extractor",
                executable="autoware_override_extractor_node",
                name="override_extractor_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "input_dir": LaunchConfiguration("input_dir"),
                        "output_dir": LaunchConfiguration("output_dir"),
                    },
                ],
            ),
        ]
    )
