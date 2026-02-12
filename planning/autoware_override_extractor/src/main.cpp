// Copyright 2026 TIER IV, Inc.
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

#include "parallel_executor.hpp"
#include "type_alias.hpp"

#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace override_event_extractor
{

class OverrideExtractorNode : public rclcpp::Node
{
public:
  OverrideExtractorNode() : Node("override_extractor_node")
  {
    declare_parameter("input_dir", "");
    declare_parameter("output_dir", "");
    declare_parameter("control_mode_topic", "/vehicle/status/control_mode");
    declare_parameter("override_mode_value", 4);
    declare_parameter("autonomous_mode_value", 1);
    declare_parameter("pre_margin", 1.0);
    declare_parameter("post_margin", 10.0);
    declare_parameter("filter_brief_overrides", true);
    declare_parameter("min_override_duration", 0.5);
    declare_parameter("num_threads", -1);
    declare_parameter("recursive_scan", false);
    declare_parameter("output_format", "mcap");
    declare_parameter("storage_preset", "mcap");
    declare_parameter(
      "preserved_topics",
      std::vector<std::string>{"/vehicle/status/control_mode", "/localization/kinematic_state",
                               "/localization/acceleration",
                               "/perception/object_recognition/tracking/objects",
                               "/perception/traffic_light_recognition/traffic_signals",
                               "/vehicle/status/turn_indicators_status",
                               "/planning/mission_planning/route", "/tf", "/tf_static",
                               "/planning/trajectory", "/vehicle/status/steering_status",
                               "/vehicle/status/velocity_status"});
    declare_parameter("preserve_all_topics", false);
  }

  ExecutorConfig get_config() const
  {
    ExecutorConfig config;

    config.input_dir = get_parameter("input_dir").as_string();
    config.output_dir = get_parameter("output_dir").as_string();
    config.num_threads = get_parameter("num_threads").as_int();
    config.recursive = get_parameter("recursive_scan").as_bool();

    config.processor_config.detector_config.control_mode_topic =
      get_parameter("control_mode_topic").as_string();
    config.processor_config.detector_config.override_mode_value =
      get_parameter("override_mode_value").as_int();
    config.processor_config.detector_config.autonomous_mode_value =
      get_parameter("autonomous_mode_value").as_int();
    config.processor_config.detector_config.pre_margin_sec = get_parameter("pre_margin").as_double();
    config.processor_config.detector_config.post_margin_sec =
      get_parameter("post_margin").as_double();
    config.processor_config.detector_config.filter_brief_overrides =
      get_parameter("filter_brief_overrides").as_bool();
    config.processor_config.detector_config.min_override_duration_sec =
      get_parameter("min_override_duration").as_double();

    config.processor_config.splitter_config.preserved_topics =
      get_parameter("preserved_topics").as_string_array();
    config.processor_config.splitter_config.preserve_all_topics =
      get_parameter("preserve_all_topics").as_bool();
    config.processor_config.splitter_config.storage_id = get_parameter("output_format").as_string();
    config.processor_config.splitter_config.serialization_format = "cdr";
    config.processor_config.splitter_config.route_topic = "/planning/mission_planning/route";

    return config;
  }
};

void print_usage(const char * program_name)
{
  std::cout << "Usage: " << program_name << " [options]\n"
            << "\nRequired:\n"
            << "  --input-dir <path>       Input directory containing rosbags\n"
            << "  --output-dir <path>      Output directory for extracted segments\n"
            << "\nOptional:\n"
            << "  --threads <N>            Number of parallel threads (default: auto-detect)\n"
            << "  --recursive              Scan subdirectories recursively\n"
            << "  --min-duration <sec>     Minimum override duration in seconds (default: 0.5)\n"
            << "  --no-filter-brief        Disable filtering of brief overrides\n"
            << "  --topics <t1,t2,...>     Comma-separated list of topics to preserve\n"
            << "  --all-topics             Preserve all topics (disables topic filtering)\n"
            << "  --pre-margin <sec>       Pre-margin in seconds (default: 1.0)\n"
            << "  --post-margin <sec>      Post-margin in seconds (default: 10.0)\n"
            << "  --help                   Show this help message\n"
            << std::endl;
}

std::vector<std::string> parse_topic_list(const std::string & topics_str)
{
  std::vector<std::string> topics;
  std::stringstream ss(topics_str);
  std::string topic;

  while (std::getline(ss, topic, ',')) {
    if (!topic.empty()) {
      topics.push_back(topic);
    }
  }

  return topics;
}

}  // namespace override_event_extractor

int main(int argc, char ** argv)
{
  using namespace override_event_extractor;

  rclcpp::init(argc, argv);

  auto node = std::make_shared<OverrideExtractorNode>();

  ExecutorConfig config = node->get_config();

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      rclcpp::shutdown();
      return 0;
    } else if (arg == "--input-dir" && i + 1 < argc) {
      config.input_dir = argv[++i];
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      config.num_threads = std::stoi(argv[++i]);
    } else if (arg == "--recursive") {
      config.recursive = true;
    } else if (arg == "--min-duration" && i + 1 < argc) {
      config.processor_config.detector_config.min_override_duration_sec = std::stod(argv[++i]);
    } else if (arg == "--no-filter-brief") {
      config.processor_config.detector_config.filter_brief_overrides = false;
    } else if (arg == "--topics" && i + 1 < argc) {
      config.processor_config.splitter_config.preserved_topics = parse_topic_list(argv[++i]);
    } else if (arg == "--all-topics") {
      config.processor_config.splitter_config.preserve_all_topics = true;
    } else if (arg == "--pre-margin" && i + 1 < argc) {
      config.processor_config.detector_config.pre_margin_sec = std::stod(argv[++i]);
    } else if (arg == "--post-margin" && i + 1 < argc) {
      config.processor_config.detector_config.post_margin_sec = std::stod(argv[++i]);
    } else if (arg.rfind("--ros-args", 0) == 0 || arg.rfind("-r", 0) == 0 || arg.rfind("__", 0) == 0 || arg == "--params-file" || arg == "-p") {
      if ((arg == "--params-file" || arg == "-p") && i + 1 < argc) {
        ++i;  // Skip the next argument (the file path or param assignment)
      }
      continue;
    } else if (i > 1) {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      rclcpp::shutdown();
      return 1;
    }
  }

  if (config.input_dir.empty() || config.output_dir.empty()) {
    RCLCPP_ERROR(node->get_logger(), "input_dir and output_dir parameters are required");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Override Event Extractor");
  RCLCPP_INFO(node->get_logger(), "========================");
  RCLCPP_INFO(node->get_logger(), "Input directory: %s", config.input_dir.c_str());
  RCLCPP_INFO(node->get_logger(), "Output directory: %s", config.output_dir.c_str());
  RCLCPP_INFO(
    node->get_logger(), "Pre-margin: %.1fs", config.processor_config.detector_config.pre_margin_sec);
  RCLCPP_INFO(
    node->get_logger(), "Post-margin: %.1fs",
    config.processor_config.detector_config.post_margin_sec);
  RCLCPP_INFO(
    node->get_logger(), "Min duration: %.1fs",
    config.processor_config.detector_config.min_override_duration_sec);
  RCLCPP_INFO(
    node->get_logger(), "Filter brief overrides: %s",
    config.processor_config.detector_config.filter_brief_overrides ? "yes" : "no");
  RCLCPP_INFO(node->get_logger(), "Recursive scan: %s", config.recursive ? "yes" : "no");

  try {
    ParallelExecutor executor(config);
    executor.execute();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "Fatal error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
