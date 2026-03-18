// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../perception_replayer/help_utils.hpp"
#include "trajectory_replayer.hpp"

#include <QApplication>
#include <QCommandLineParser>
#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using autoware::planning_debug_tools::trajectory_replayer::ReplayMode;
using autoware::planning_debug_tools::trajectory_replayer::TrajectoryReplayer;
using autoware::planning_debug_tools::trajectory_replayer::TrajectoryReplayerParam;

int main(int argc, char * argv[])
{
  try {
    std::vector<std::string> non_ros_args = rclcpp::init_and_remove_ros_arguments(argc, argv);
    std::vector<char *> qt_argv;
    qt_argv.push_back(argv[0]);
    for (auto & arg : non_ros_args) {
      qt_argv.push_back(const_cast<char *>(arg.c_str()));
    }
    int qt_argc = static_cast<int>(qt_argv.size());

    QApplication app(qt_argc, qt_argv.data());
    app.setApplicationName("trajectory_replayer");
    app.setApplicationVersion(PACKAGE_VERSION);
    const std::string app_desc =
      "Replays trajectory-related topics from a rosbag through live modifier/optimizer nodes.\n"
      "\n"
      "Use with the trajectory_replayer.launch.xml to start the modifier and optimizer nodes,\n"
      "then run this tool to control playback with a time slider.\n"
      "\n"
      "Modes:\n"
      "  modifier  - Replay diffusion planner output + perception. Live modifier + optimizer.\n"
      "  optimizer - Replay modifier output directly. Live optimizer only.\n";

    QCommandLineParser parser;

    QList<QCommandLineOption> options;
    QCommandLineOption help_option(
      QStringList() << "h"
                    << "help",
      "show this help message and exit");
    options << help_option;

    QCommandLineOption bag_option(
      QStringList() << "b"
                    << "bag",
      "rosbag path", "BAG");
    options << bag_option;

    QCommandLineOption format_option(
      QStringList() << "f"
                    << "rosbag-format",
      "rosbag data format", "{sqlite3,mcap}", "mcap");
    options << format_option;

    QCommandLineOption mode_option(
      QStringList() << "m"
                    << "mode",
      "replay mode: modifier (full pipeline) or optimizer (optimizer only)", "{modifier,optimizer}",
      "optimizer");
    options << mode_option;

    QCommandLineOption pointcloud_option(
      QStringList() << "p"
                    << "replay-pointcloud",
      "replay pointcloud data (modifier mode only, uses more memory)");
    options << pointcloud_option;

    for (const auto & option : options) {
      parser.addOption(option);
    }

    parser.process(app);

    if (parser.isSet(help_option)) {
      show_help(options, "ros2 run planning_debug_tools trajectory_replayer", app_desc);
      return 0;
    }

    TrajectoryReplayerParam param;
    param.rosbag_path = parser.value(bag_option).toStdString();

    if (param.rosbag_path.empty()) {
      std::cerr << "Error: bag path is required." << std::endl << std::endl;
      show_help(options, "ros2 run planning_debug_tools trajectory_replayer", app_desc);
      return 1;
    }

    param.rosbag_format = parser.value(format_option).toStdString();
    if (param.rosbag_format != "sqlite3" && param.rosbag_format != "mcap") {
      std::cerr << "Error: invalid rosbag format: " << param.rosbag_format << std::endl;
      return 1;
    }

    const std::string mode_str = parser.value(mode_option).toStdString();
    if (mode_str == "modifier") {
      param.mode = ReplayMode::MODIFIER;
    } else if (mode_str == "optimizer") {
      param.mode = ReplayMode::OPTIMIZER;
    } else {
      std::cerr << "Error: invalid mode: " << mode_str << std::endl;
      return 1;
    }

    param.replay_pointcloud = parser.isSet(pointcloud_option);

    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<TrajectoryReplayer>(param, node_options);
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    while (rclcpp::ok()) {
      executor.spin_some();
      app.processEvents();
    }
    rclcpp::shutdown();
  } catch (const std::exception & e) {
    std::cerr << "Exception in main(): " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception in main()" << std::endl;
    return 1;
  }
  return 0;
}
