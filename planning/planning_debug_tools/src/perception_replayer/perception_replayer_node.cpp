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

#include "help_utils.hpp"
#include "perception_replayer.hpp"

#include <QApplication>
#include <QCommandLineParser>
#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using autoware::planning_debug_tools::PerceptionReplayer;
using autoware::planning_debug_tools::PerceptionReplayerCommonParam;

int main(int argc, char * argv[])
{
  try {
    // Create arguments for Qt application
    std::vector<std::string> non_ros_args = rclcpp::init_and_remove_ros_arguments(argc, argv);
    std::vector<char *> qt_argv;
    qt_argv.push_back(argv[0]);
    for (auto & arg : non_ros_args) {
      qt_argv.push_back(const_cast<char *>(arg.c_str()));
    }
    int qt_argc = static_cast<int>(qt_argv.size());

    QApplication app(qt_argc, qt_argv.data());
    app.setApplicationName("perception_replayer");
    app.setApplicationVersion(PACKAGE_VERSION);
    const std::string app_desc =
      "This script can overlay the perception results from the rosbag on the planning simulator.\n"
      "\n"
      "In detail, this script publishes the data at a certain timestamp from the rosbag.\n"
      "The timestamp will increase according to the real time without any operation.\n"
      "By using the GUI, you can modify the timestamp by pausing, changing the rate or going back "
      "into the past.";

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
      "rosbag", "BAG");
    options << bag_option;

    QCommandLineOption tracked_object_option(
      QStringList() << "t"
                    << "tracked-object",
      "publish tracked object");
    options << tracked_object_option;

    QCommandLineOption rosbag_format_option(
      QStringList() << "f"
                    << "rosbag-format",
      "rosbag data format", "{sqlite3,mcap}", "sqlite3");
    options << rosbag_format_option;

    QCommandLineOption enable_cache_option(
      QStringList() << "cache",
      "enable cache-based loading instead of loading all rosbag data at once");
    options << enable_cache_option;

    QCommandLineOption cache_window_before_option(
      QStringList() << "cache-window-before",
      "cache window size before current timestamp (seconds)", "seconds", "30.0");
    options << cache_window_before_option;

    QCommandLineOption cache_window_option(
      QStringList() << "cache-window-after",
      "cache window size after current timestamp (seconds)", "seconds", "60.0");
    options << cache_window_option;

    for (const auto & option : options) {
      parser.addOption(option);
    }

    parser.process(app);

    // Check if help was requested
    if (parser.isSet(help_option)) {
      show_help(options, "ros2 run planning_debug_tools perception_replayer", app_desc);
      return 0;
    }

    // Validate rosbag path
    const std::string rosbag_path = parser.value(bag_option).toStdString();
    if (rosbag_path.empty()) {
      std::cerr << "Error: bag path is required." << std::endl << std::endl;
      show_help(options, "ros2 run planning_debug_tools perception_replayer", app_desc);
      return 1;
    }

    // Validate rosbag format
    const std::string rosbag_format = parser.value(rosbag_format_option).toStdString();
    if (rosbag_format != "sqlite3" && rosbag_format != "mcap") {
      std::cerr << "Error: invalid rosbag format: " << rosbag_format << std::endl;
      return 1;
    }

    // Initialize parameters
    PerceptionReplayerCommonParam perception_replayer_param{
      .tracked_object = parser.isSet(tracked_object_option),
      .use_rosbag_route = false  // not used for replayer
    };

    // Initialize rosbag manager
    RosbagManagerParam rosbag_manager_param{
      .rosbag_format = rosbag_format,
      .enable_cache = parser.isSet(enable_cache_option),
      .cache_window_before_sec = parser.value(cache_window_before_option).toDouble(),
      .cache_window_after_sec = parser.value(cache_window_option).toDouble()
    };
    auto rosbag_manager = std::make_unique<autoware::planning_debug_tools::RosbagManager>(
      rclcpp::get_logger("rosbag_manager"), perception_replayer_param.tracked_object, rosbag_manager_param);
    rosbag_manager->initialize_from_path(rosbag_path);

    rclcpp::NodeOptions node_options;
    auto node = std::make_shared<PerceptionReplayer>(
      perception_replayer_param, std::move(rosbag_manager), node_options);

    while (rclcpp::ok()) {
      rclcpp::spin_some(node);
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
