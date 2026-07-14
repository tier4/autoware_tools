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
#include "perception_reproducer.hpp"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
  try {
    std::vector<std::string> non_ros_args = rclcpp::init_and_remove_ros_arguments(argc, argv);

    std::vector<char *> qt_argv;
    qt_argv.push_back(argv[0]);
    for (auto & arg : non_ros_args) {
      qt_argv.push_back(const_cast<char *>(arg.c_str()));
    }
    int qt_argc = static_cast<int>(qt_argv.size());

    QCoreApplication app(qt_argc, qt_argv.data());
    app.setApplicationName("perception_reproducer");
    app.setApplicationVersion(PACKAGE_VERSION);
    const std::string app_desc =
      "Replay perception outputs from rosbag data to reproduce planning behaviour.";

    QCommandLineParser parser;

    QList<QCommandLineOption> options;
    const QCommandLineOption help_option(
      QStringList() << "h"
                    << "help",
      "show this help message and exit");
    options << help_option;

    const QCommandLineOption bag_option(
      QStringList() << "b"
                    << "bag",
      "rosbag", "BAG");
    options << bag_option;

    const QCommandLineOption tracked_object_option(
      QStringList() << "t"
                    << "tracked-object",
      "publish tracked object");
    options << tracked_object_option;

    const QCommandLineOption noise_option(
      QStringList() << "n"
                    << "noise",
      "apply perception noise to the objects when publishing repeated messages");
    options << noise_option;

    const QCommandLineOption rosbag_format_option(
      QStringList() << "f"
                    << "rosbag-format",
      "rosbag data format", "{sqlite3,mcap}", "sqlite3");
    options << rosbag_format_option;

    const QCommandLineOption search_radius_option(
      QStringList() << "r"
                    << "search-radius",
      "the search radius for searching rosbag's ego odom messages around the nearest ego odom pose "
      "if the search radius is set to 0, it will always publish the "
      "closest message, just as the old reproducer did.",
      "radius", "1.5");
    options << search_radius_option;

    const QCommandLineOption cool_down_option(
      QStringList() << "c"
                    << "reproduce-cool-down",
      "The cool down time for republishing published messages, please "
      "make sure that it's greater than the ego's stopping time.",
      "seconds", "80.0");
    options << cool_down_option;

    const QCommandLineOption pub_route_option(
      QStringList() << "p"
                    << "pub-route",
      "publish route created from the initial pose and goal pose retrieved from the beginning and "
      "end of the rosbag. By default, route are not published.");
    options << pub_route_option;

    const QCommandLineOption replay_route_option(
      QStringList() << "replay-route",
      "replay route and route state topics from rosbag data with transient_local QoS. "
      "Only publishes when the message changes (i.e., not redundantly).");
    options << replay_route_option;

    const QCommandLineOption verbose_option(
      QStringList() << "v"
                    << "verbose",
      "output debug data.");
    options << verbose_option;

    const QCommandLineOption ref_image_topics_option(
      QStringList() << "reference-image-topics",
      "comma-separated list of CompressedImage topics to load and publish "
      "(e.g., "
      "'/sensing/camera/camera0/image_raw/compressed,/sensing/camera/camera1/image_raw/"
      "compressed'). "
      "Each topic will be loaded from rosbag and published to the same topic name.",
      "topics", "");
    options << ref_image_topics_option;

    const QCommandLineOption auto_tackle_stuck_option(
      QStringList() << "auto-tackle-stuck",
      "when ego is stopped and reproducer stays in repeat longer than --stuck-duration, "
      "first rebuild with search radius scaled by --expand-radius-scale; if still stuck for "
      "another --stuck-duration, call /localization/initialize (DIRECT) with a pose further "
      "along the bag trajectory");
    options << auto_tackle_stuck_option;

    const QCommandLineOption stuck_duration_option(
      QStringList() << "stuck-duration",
      "seconds that ego must be stopped while reproducer is in repeat before tackle steps "
      "(expand, then perturb)",
      "seconds", "5.0");
    options << stuck_duration_option;

    const QCommandLineOption expand_radius_scale_option(
      QStringList() << "expand-radius-scale",
      "scale applied to --search-radius for the one-shot stuck expand rebuild", "scale", "3.0");
    options << expand_radius_scale_option;

    const QCommandLineOption perturb_distance_option(
      QStringList() << "perturb-distance",
      "path distance [m] to walk forward along bag ego poses when perturbing", "meters", "0.5");
    options << perturb_distance_option;

    const QCommandLineOption output_metrics_option(
      QStringList() << "output-metrics",
      "dump reproducer metrics json under the ROS logging directory on shutdown");
    options << output_metrics_option;

    for (const auto & option : options) {
      parser.addOption(option);
    }

    parser.process(app);

    if (parser.isSet(help_option)) {
      show_help(options, "ros2 run planning_debug_tools perception_reproducer", app_desc);
      return 0;
    }

    if (!parser.isSet(bag_option)) {
      std::cerr << "Error: bag path is required." << std::endl << std::endl;
      show_help(options, "ros2 run planning_debug_tools perception_reproducer", app_desc);
      return 1;
    }

    autoware::planning_debug_tools::PerceptionReproducerParam param;
    param.rosbag_path = parser.value(bag_option).toStdString();
    param.rosbag_format = parser.value(rosbag_format_option).toStdString();
    param.tracked_object = parser.isSet(tracked_object_option);
    param.search_radius = parser.value(search_radius_option).toDouble();
    param.reproduce_cool_down = parser.value(cool_down_option).toDouble();
    param.noise = parser.isSet(noise_option);
    param.verbose = parser.isSet(verbose_option);
    param.publish_route = parser.isSet(pub_route_option);
    param.replay_route = parser.isSet(replay_route_option);
    param.auto_tackle_stuck = parser.isSet(auto_tackle_stuck_option);
    param.stuck_duration_s = parser.value(stuck_duration_option).toDouble();
    param.expand_radius_scale = parser.value(expand_radius_scale_option).toDouble();
    param.perturb_distance_m = parser.value(perturb_distance_option).toDouble();
    param.output_metrics = parser.isSet(output_metrics_option);

    if (param.stuck_duration_s < 0.0) {
      std::cerr << "Error: --stuck-duration must be >= 0" << std::endl;
      return 1;
    }
    if (param.expand_radius_scale <= 0.0) {
      std::cerr << "Error: --expand-radius-scale must be > 0" << std::endl;
      return 1;
    }
    if (param.perturb_distance_m < 0.0) {
      std::cerr << "Error: --perturb-distance must be >= 0" << std::endl;
      return 1;
    }

    // Parse comma-separated reference image topics
    const QString topics_str = parser.value(ref_image_topics_option);
    if (!topics_str.isEmpty()) {
      const QStringList topic_list = topics_str.split(',', Qt::SkipEmptyParts);
      for (const auto & topic : topic_list) {
        param.reference_image_topics.push_back(topic.trimmed().toStdString());
      }
      std::cout << "Reference image topics: " << param.reference_image_topics.size()
                << " topics configured" << std::endl;
    }

    if (param.rosbag_format != "sqlite3" && param.rosbag_format != "mcap") {
      std::cerr << "Error: invalid rosbag format: " << param.rosbag_format << std::endl;
      return 1;
    }

    auto node = std::make_shared<autoware::planning_debug_tools::PerceptionReproducer>(
      param, rclcpp::NodeOptions());
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();
    rclcpp::shutdown();
  } catch (const std::exception & e) {
    std::cerr << "Exception in main(): " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception in main()" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  return 0;
}
