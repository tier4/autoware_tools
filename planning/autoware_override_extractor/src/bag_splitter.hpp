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

#ifndef BAG_SPLITTER_HPP_
#define BAG_SPLITTER_HPP_

#include "type_alias.hpp"

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_cpp/writers/sequential_writer.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include <string>
#include <unordered_set>
#include <vector>

namespace override_event_extractor
{

class BagSplitter
{
public:
  explicit BagSplitter(const SplitterConfig & config);

  std::vector<std::string> split(
    const std::string & input_bag, const std::vector<OverrideEvent> & events,
    const std::string & output_dir, const RouteMessage & route_msg = RouteMessage());

private:
  void extractSegment(
    const std::string & input_bag, const OverrideEvent & event, const std::string & output_path,
    const RouteMessage & route_msg);

  bool inRange(int64_t timestamp, const TimeRange & range) const;

  bool shouldPreserveTopic(const std::string & topic_name) const;

  std::string getBagBaseName(const std::string & bag_path) const;

  SplitterConfig config_;
  std::unordered_set<std::string> preserved_topics_set_;
};

}  // namespace override_event_extractor

#endif  // BAG_SPLITTER_HPP_
