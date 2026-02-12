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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace override_event_extractor
{

void printUsage(const char * program_name)
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
            << "  --pre-margin <sec>       Pre-margin in seconds (default: 1.0)\n"
            << "  --post-margin <sec>      Post-margin in seconds (default: 10.0)\n"
            << "  --help                   Show this help message\n"
            << std::endl;
}

std::vector<std::string> parseTopicList(const std::string & topics_str)
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

  ExecutorConfig config;

  config.processor_config.detector_config.control_mode_topic = "/vehicle/status/control_mode";
  config.processor_config.detector_config.override_mode_value = 4;
  config.processor_config.detector_config.autonomous_mode_value = 1;
  config.processor_config.detector_config.pre_margin_sec = 1.0;
  config.processor_config.detector_config.post_margin_sec = 10.0;
  config.processor_config.detector_config.filter_brief_overrides = true;
  config.processor_config.detector_config.min_override_duration_sec = 0.5;

  config.processor_config.splitter_config.preserved_topics = {
    "/vehicle/status/control_mode", "/localization/kinematic_state",
    "/perception/object_recognition/objects", "/planning/trajectory",
    "/vehicle/status/steering_status", "/vehicle/status/velocity_status", "/tf", "/tf_static",
    "/planning/mission_planning/route"};
  config.processor_config.splitter_config.storage_id = "mcap";
  config.processor_config.splitter_config.serialization_format = "cdr";
  config.processor_config.splitter_config.route_topic = "/planning/mission_planning/route";

  config.num_threads = -1;
  config.recursive = false;

  bool has_input_dir = false;
  bool has_output_dir = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--input-dir" && i + 1 < argc) {
      config.input_dir = argv[++i];
      has_input_dir = true;
    } else if (arg == "--output-dir" && i + 1 < argc) {
      config.output_dir = argv[++i];
      has_output_dir = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      config.num_threads = std::stoi(argv[++i]);
    } else if (arg == "--recursive") {
      config.recursive = true;
    } else if (arg == "--min-duration" && i + 1 < argc) {
      config.processor_config.detector_config.min_override_duration_sec = std::stod(argv[++i]);
    } else if (arg == "--no-filter-brief") {
      config.processor_config.detector_config.filter_brief_overrides = false;
    } else if (arg == "--topics" && i + 1 < argc) {
      config.processor_config.splitter_config.preserved_topics = parseTopicList(argv[++i]);
    } else if (arg == "--pre-margin" && i + 1 < argc) {
      config.processor_config.detector_config.pre_margin_sec = std::stod(argv[++i]);
    } else if (arg == "--post-margin" && i + 1 < argc) {
      config.processor_config.detector_config.post_margin_sec = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  }

  if (!has_input_dir || !has_output_dir) {
    std::cerr << "Error: --input-dir and --output-dir are required" << std::endl;
    printUsage(argv[0]);
    return 1;
  }

  std::cout << "Override Event Extractor" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << "Input directory: " << config.input_dir << std::endl;
  std::cout << "Output directory: " << config.output_dir << std::endl;
  std::cout << "Pre-margin: " << config.processor_config.detector_config.pre_margin_sec << "s"
            << std::endl;
  std::cout << "Post-margin: " << config.processor_config.detector_config.post_margin_sec << "s"
            << std::endl;
  std::cout << "Min duration: " << config.processor_config.detector_config.min_override_duration_sec
            << "s" << std::endl;
  std::cout << "Filter brief overrides: "
            << (config.processor_config.detector_config.filter_brief_overrides ? "yes" : "no")
            << std::endl;
  std::cout << "Recursive scan: " << (config.recursive ? "yes" : "no") << std::endl;
  std::cout << std::endl;

  try {
    ParallelExecutor executor(config);
    executor.execute();
  } catch (const std::exception & e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
