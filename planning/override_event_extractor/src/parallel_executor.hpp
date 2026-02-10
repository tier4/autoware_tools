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

#ifndef PARALLEL_EXECUTOR_HPP_
#define PARALLEL_EXECUTOR_HPP_

#include "bag_processor.hpp"
#include "type_alias.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace override_event_extractor
{

class ParallelExecutor
{
public:
  explicit ParallelExecutor(const ExecutorConfig & config);

  BatchResult execute();

private:
  std::vector<std::string> discoverRosbags() const;

  void initializeStoragePlugins(const std::vector<std::string> & rosbags);

  std::string getBagSeriesName(const std::string & bag_path) const;

  void extractRouteMessages(const std::vector<std::string> & rosbags);

  void processWithThreadPool(
    const std::vector<std::string> & bags, std::vector<ProcessResult> & results);

  void generateBatchSummary(const BatchResult & result) const;

  ExecutorConfig config_;
  BagProcessor processor_;
  std::unordered_map<std::string, RouteMessage> route_cache_;
};

}  // namespace override_event_extractor

#endif  // PARALLEL_EXECUTOR_HPP_
