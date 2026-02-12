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

#include <rosbag2_cpp/reader.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>
#include <unordered_set>

namespace override_event_extractor
{

ParallelExecutor::ParallelExecutor(const ExecutorConfig & config)
: config_(config), processor_(config.processor_config)
{
}

void ParallelExecutor::initialize_storage_plugins(const std::vector<std::string> & rosbags)
{
  if (rosbags.empty()) {
    return;
  }

  // Open and close several bags to ensure plugins are fully loaded
  // This prevents race conditions when many threads start simultaneously
  int bags_to_open = std::min(static_cast<int>(rosbags.size()), 3);
  for (int i = 0; i < bags_to_open; ++i) {
    try {
      rosbag2_cpp::Reader reader;
      reader.open(rosbags[i]);
      // Give plugin time to fully initialize
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } catch (const std::exception & e) {
      // Ignore errors during pre-initialization
    }
  }

  // Extract route messages from _0 bags
  extract_route_messages(rosbags);
}

std::string ParallelExecutor::get_bag_series_name(const std::string & bag_path) const
{
  std::filesystem::path p(bag_path);
  std::string stem = p.stem().string();

  // Remove _N suffix (e.g., recording_name_0 -> recording_name)
  auto pos = stem.rfind('_');
  if (pos != std::string::npos) {
    std::string suffix = stem.substr(pos + 1);
    // Check if suffix is a number
    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
      return stem.substr(0, pos);
    }
  }
  return stem;
}

void ParallelExecutor::extract_route_messages(const std::vector<std::string> & rosbags)
{
  const std::string route_topic = config_.processor_config.splitter_config.route_topic;
  std::cout << "Extracting route messages from _0 bags..." << std::endl;

  // Find all _0 bags first
  std::vector<std::string> zero_bags;
  std::unordered_set<std::string> seen_series;

  for (const auto & bag_path : rosbags) {
    std::filesystem::path p(bag_path);
    std::string stem = p.stem().string();

    // Only process _0 bags
    if (stem.size() < 2 || stem.substr(stem.size() - 2) != "_0") {
      continue;
    }

    std::string series_name = get_bag_series_name(bag_path);

    // Skip if already seen this series
    if (seen_series.find(series_name) != seen_series.end()) {
      continue;
    }

    seen_series.insert(series_name);
    zero_bags.push_back(bag_path);
  }

  if (zero_bags.empty()) {
    std::cout << "No _0 bags found" << std::endl;
    return;
  }

  // Extract routes sequentially (fast enough, avoids plugin race conditions)
  for (const auto & bag_path : zero_bags) {
    std::string series_name = get_bag_series_name(bag_path);

    try {
      rosbag2_cpp::Reader reader;
      reader.open(bag_path);

      rosbag2_storage::StorageFilter filter;
      filter.topics.push_back(route_topic);
      reader.set_filter(filter);

      if (reader.has_next()) {
        auto bag_message = reader.read_next();
        if (bag_message->topic_name == route_topic) {
          RouteMessage route_msg;
          route_msg.message = std::make_shared<rosbag2_storage::SerializedBagMessage>(*bag_message);
          route_msg.valid = true;
          route_cache_[series_name] = route_msg;

          std::filesystem::path p(bag_path);
          std::cout << "  Extracted route from: " << p.filename().string() << std::endl;
        }
      }
    } catch (const std::exception & e) {
      std::filesystem::path p(bag_path);
      std::cerr << "  Warning: Could not extract route from " << p.filename().string() << ": "
                << e.what() << std::endl;
    }
  }

  std::cout << "Found " << route_cache_.size() << " route message(s)" << std::endl;
}

BatchResult ParallelExecutor::execute()
{
  BatchResult batch_result;

  std::filesystem::create_directories(config_.output_dir);

  auto rosbags = discover_rosbags();
  std::cout << "Discovered " << rosbags.size() << " rosbag(s) to process" << std::endl;

  if (rosbags.empty()) {
    return batch_result;
  }

  // Pre-initialize storage plugins to avoid race conditions in threads
  initialize_storage_plugins(rosbags);

  std::vector<ProcessResult> results(rosbags.size());
  process_with_thread_pool(rosbags, results);

  batch_result.results = results;
  batch_result.total_bags_processed = rosbags.size();

  for (const auto & result : results) {
    if (result.success) {
      batch_result.total_overrides_found += result.num_overrides_found;
    } else {
      batch_result.failed_bags++;
    }
  }

  generate_batch_summary(batch_result);

  std::cout << "\nBatch processing complete:" << std::endl;
  std::cout << "  Total bags processed: " << batch_result.total_bags_processed << std::endl;
  std::cout << "  Total overrides found: " << batch_result.total_overrides_found << std::endl;
  std::cout << "  Failed bags: " << batch_result.failed_bags << std::endl;

  return batch_result;
}

std::vector<std::string> ParallelExecutor::discover_rosbags() const
{
  std::vector<std::string> rosbags;

  if (!std::filesystem::exists(config_.input_dir)) {
    std::cerr << "Input directory does not exist: " << config_.input_dir << std::endl;
    return rosbags;
  }

  auto search_dir = [&rosbags](const std::filesystem::path & dir) {
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
      if (entry.is_regular_file()) {
        const auto ext = entry.path().extension().string();
        if (ext == ".mcap" || ext == ".db3") {
          rosbags.push_back(entry.path().string());
        }
      }
    }
  };

  if (config_.recursive) {
    for (const auto & entry : std::filesystem::recursive_directory_iterator(config_.input_dir)) {
      if (entry.is_regular_file()) {
        const auto ext = entry.path().extension().string();
        if (ext == ".mcap" || ext == ".db3") {
          rosbags.push_back(entry.path().string());
        }
      }
    }
  } else {
    search_dir(config_.input_dir);
  }

  return rosbags;
}

void ParallelExecutor::process_with_thread_pool(
  const std::vector<std::string> & bags, std::vector<ProcessResult> & results)
{
  int num_threads = config_.num_threads;
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
  }

  std::cout << "Using " << num_threads << " thread(s)" << std::endl;

  std::vector<std::future<ProcessResult>> futures;
  std::vector<size_t> skipped_indices;

  for (size_t i = 0; i < bags.size(); ++i) {
    // Look up route for this bag's series
    std::string series_name = get_bag_series_name(bags[i]);
    auto it = route_cache_.find(series_name);

    if (it == route_cache_.end() || !it->second.valid) {
      // No route found for this series - skip the bag
      std::filesystem::path p(bags[i]);
      std::cerr << "Warning: Skipping " << p.filename().string()
                << " - no route found for series '" << series_name << "'" << std::endl;
      ProcessResult skipped_result;
      skipped_result.input_bag = bags[i];
      skipped_result.success = false;
      skipped_result.error_message = "No route message found for bag series '" + series_name + "'";
      results[i] = skipped_result;
      continue;
    }

    RouteMessage route_msg = it->second;

    auto future = std::async(
      std::launch::async, [this, &bags, i, route_msg]() {
        return processor_.process(bags[i], config_.output_dir, route_msg);
      });
    futures.push_back(std::move(future));

    if (futures.size() >= static_cast<size_t>(num_threads)) {
      for (auto & f : futures) {
        results[i - futures.size() + 1 + (&f - &futures[0])] = f.get();
      }
      futures.clear();
    }
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    results[bags.size() - futures.size() + i] = futures[i].get();
  }
}

void ParallelExecutor::generate_batch_summary(const BatchResult & result) const
{
  const auto summary_path = std::filesystem::path(config_.output_dir) / "batch_summary.json";
  std::ofstream out(summary_path);

  if (!out.is_open()) {
    std::cerr << "Failed to create batch summary file: " << summary_path << std::endl;
    return;
  }

  out << "{\n";
  out << "  \"input_directory\": \"" << config_.input_dir << "\",\n";
  out << "  \"output_directory\": \"" << config_.output_dir << "\",\n";
  out << "  \"configuration\": {\n";
  out << "    \"num_threads\": " << config_.num_threads << ",\n";
  out << "    \"pre_margin_sec\": " << config_.processor_config.detector_config.pre_margin_sec
      << ",\n";
  out << "    \"post_margin_sec\": " << config_.processor_config.detector_config.post_margin_sec
      << ",\n";
  out << "    \"min_override_duration_sec\": "
      << config_.processor_config.detector_config.min_override_duration_sec << ",\n";
  out << "    \"filter_brief_overrides\": "
      << (config_.processor_config.detector_config.filter_brief_overrides ? "true" : "false")
      << "\n";
  out << "  },\n";
  out << "  \"results\": {\n";
  out << "    \"total_bags_processed\": " << result.total_bags_processed << ",\n";
  out << "    \"total_overrides_found\": " << result.total_overrides_found << ",\n";
  out << "    \"failed_bags\": " << result.failed_bags << "\n";
  out << "  },\n";
  out << "  \"per_bag_summary\": [\n";

  for (size_t i = 0; i < result.results.size(); ++i) {
    const auto & r = result.results[i];
    std::filesystem::path p(r.input_bag);
    out << "    {\n";
    out << "      \"bag_name\": \"" << p.filename().string() << "\",\n";
    out << "      \"overrides_found\": " << r.num_overrides_found << ",\n";
    out << "      \"success\": " << (r.success ? "true" : "false") << "\n";
    out << "    }" << (i < result.results.size() - 1 ? ",\n" : "\n");
  }

  out << "  ]\n";
  out << "}\n";

  out.close();
}

}  // namespace override_event_extractor
