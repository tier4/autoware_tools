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

#include "bag_processor.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace override_event_extractor
{

BagProcessor::BagProcessor(const ProcessorConfig & config)
: config_(config), detector_(config.detector_config), splitter_(config.splitter_config)
{
}

ProcessResult BagProcessor::process(
  const std::string & bag_path, const std::string & output_dir, const RouteMessage & route_msg)
{
  ProcessResult result;
  result.input_bag = bag_path;

  const auto bag_name = std::filesystem::path(bag_path).filename().string();

  try {
    if (!std::filesystem::exists(bag_path)) {
      result.error_message = "Input bag does not exist: " + bag_path;
      return result;
    }

    std::cout << "Processing: " << bag_path << std::endl;

    auto events = detector_.detect(bag_path);
    result.num_overrides_found = events.size();
    result.events = events;

    if (events.empty()) {
      std::cout << "  [" << bag_name << "] No override events found." << std::endl;
      result.success = true;
      return result;
    }

    std::cout << "  [" << bag_name << "] Found " << events.size() << " override event(s)" << std::endl;

    const auto base_name = get_bag_base_name(bag_path);
    const auto date = extract_date_from_bag_name(base_name);
    const auto output_subdir = std::filesystem::path(output_dir) / date / base_name;

    std::filesystem::create_directories(output_subdir);

    result.output_files = splitter_.split(bag_path, events, output_subdir.string(), route_msg);

    std::cout << "  [" << bag_name << "] Extracted " << result.output_files.size() << " segment(s)" << std::endl;

    generate_summary(result, output_subdir.string());

    result.success = true;

  } catch (const std::exception & e) {
    result.error_message = std::string("Exception: ") + e.what();
    std::cerr << "  [" << bag_name << "] Error: " << result.error_message << std::endl;
  }

  return result;
}

void BagProcessor::generate_summary(
  const ProcessResult & result, const std::string & output_dir) const
{
  const auto summary_path = std::filesystem::path(output_dir) / "summary.json";
  std::ofstream out(summary_path);

  if (!out.is_open()) {
    std::cerr << "Failed to create summary file: " << summary_path << std::endl;
    return;
  }

  out << "{\n";
  out << "  \"input_bag\": \"" << result.input_bag << "\",\n";
  out << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
  out << "  \"num_overrides_found\": " << result.num_overrides_found << ",\n";
  out << "  \"override_events\": [\n";

  for (size_t i = 0; i < result.events.size(); ++i) {
    const auto & event = result.events[i];
    out << "    {\n";
    out << "      \"index\": " << event.index << ",\n";
    out << "      \"raw_start_time_ns\": " << event.raw_range.start_ns << ",\n";
    out << "      \"raw_end_time_ns\": " << event.raw_range.end_ns << ",\n";
    out << "      \"raw_duration_sec\": " << std::fixed << std::setprecision(3)
        << event.raw_range.duration_seconds() << ",\n";
    out << "      \"extended_start_time_ns\": " << event.extended_range.start_ns << ",\n";
    out << "      \"extended_end_time_ns\": " << event.extended_range.end_ns << ",\n";
    out << "      \"extended_duration_sec\": " << std::fixed << std::setprecision(3)
        << event.extended_range.duration_seconds() << ",\n";
    out << "      \"output_file\": \"" << result.output_files[i] << "\"\n";
    out << "    }" << (i < result.events.size() - 1 ? ",\n" : "\n");
  }

  out << "  ]\n";
  out << "}\n";

  out.close();
}

std::string BagProcessor::get_bag_base_name(const std::string & bag_path) const
{
  std::filesystem::path p(bag_path);
  auto stem = p.stem().string();

  if (stem.size() >= 2 && stem.substr(stem.size() - 2) == "_0") {
    stem = stem.substr(0, stem.length() - 2);
  }

  return stem;
}

std::string BagProcessor::extract_date_from_bag_name(const std::string & bag_name) const
{
  size_t pos = bag_name.find("_20");
  if (pos == std::string::npos) {
    return "unknown_date";
  }

  std::string date_part = bag_name.substr(pos + 1);

  if (date_part.length() >= 10) {
    return date_part.substr(0, 10);
  }

  return "unknown_date";
}

}  // namespace override_event_extractor
