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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>

namespace override_event_extractor
{

ParallelExecutor::ParallelExecutor(const ExecutorConfig & config)
: config_(config), processor_(config.processor_config)
{
}

BatchResult ParallelExecutor::execute()
{
  BatchResult batch_result;

  std::filesystem::create_directories(config_.output_dir);

  auto rosbags = discoverRosbags();
  std::cout << "Discovered " << rosbags.size() << " rosbag(s) to process" << std::endl;

  if (rosbags.empty()) {
    return batch_result;
  }

  std::vector<ProcessResult> results(rosbags.size());
  processWithThreadPool(rosbags, results);

  batch_result.results = results;
  batch_result.total_bags_processed = rosbags.size();

  for (const auto & result : results) {
    if (result.success) {
      batch_result.total_overrides_found += result.num_overrides_found;
    } else {
      batch_result.failed_bags++;
    }
  }

  generateBatchSummary(batch_result);

  std::cout << "\nBatch processing complete:" << std::endl;
  std::cout << "  Total bags processed: " << batch_result.total_bags_processed << std::endl;
  std::cout << "  Total overrides found: " << batch_result.total_overrides_found << std::endl;
  std::cout << "  Failed bags: " << batch_result.failed_bags << std::endl;

  return batch_result;
}

std::vector<std::string> ParallelExecutor::discoverRosbags() const
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

void ParallelExecutor::processWithThreadPool(
  const std::vector<std::string> & bags, std::vector<ProcessResult> & results)
{
  int num_threads = config_.num_threads;
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
  }

  std::cout << "Using " << num_threads << " thread(s)" << std::endl;

  std::vector<std::future<ProcessResult>> futures;

  for (size_t i = 0; i < bags.size(); ++i) {
    auto future = std::async(
      std::launch::async,
      [this, &bags, i]() { return processor_.process(bags[i], config_.output_dir); });
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

void ParallelExecutor::generateBatchSummary(const BatchResult & result) const
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
