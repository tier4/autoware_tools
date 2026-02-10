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

#ifndef TYPE_ALIAS_HPP_
#define TYPE_ALIAS_HPP_

#include <autoware_vehicle_msgs/msg/control_mode_report.hpp>

#include <rosbag2_storage/bag_metadata.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace override_event_extractor
{

using autoware_vehicle_msgs::msg::ControlModeReport;

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::seconds;

struct TimeRange
{
  int64_t start_ns;
  int64_t end_ns;

  double duration_seconds() const { return (end_ns - start_ns) / 1e9; }

  bool overlaps(const TimeRange & other) const
  {
    return !(end_ns < other.start_ns || start_ns > other.end_ns);
  }

  TimeRange merge(const TimeRange & other) const
  {
    return TimeRange{std::min(start_ns, other.start_ns), std::max(end_ns, other.end_ns)};
  }
};

struct OverrideEvent
{
  TimeRange raw_range;
  TimeRange extended_range;
  size_t index;

  OverrideEvent(const TimeRange & raw, const TimeRange & extended, size_t idx)
  : raw_range(raw), extended_range(extended), index(idx)
  {
  }
};

struct DetectorConfig
{
  std::string control_mode_topic{"/vehicle/status/control_mode"};
  uint8_t override_mode_value{4};
  uint8_t autonomous_mode_value{1};
  double pre_margin_sec{1.0};
  double post_margin_sec{10.0};
  bool filter_brief_overrides{true};
  double min_override_duration_sec{0.5};
};

struct RouteMessage
{
  std::shared_ptr<rosbag2_storage::SerializedBagMessage> message;
  bool valid{false};
};

struct SplitterConfig
{
  std::vector<std::string> preserved_topics;
  std::string storage_id{"mcap"};
  std::string serialization_format{"cdr"};
  std::string route_topic{"/planning/mission_planning/route"};
  RouteMessage route_message;
};

struct ProcessorConfig
{
  DetectorConfig detector_config;
  SplitterConfig splitter_config;
};

struct ProcessResult
{
  std::string input_bag;
  bool success{false};
  std::string error_message;
  size_t num_overrides_found{0};
  std::vector<std::string> output_files;
  std::vector<OverrideEvent> events;
};

struct ExecutorConfig
{
  std::string input_dir;
  std::string output_dir;
  int num_threads{-1};
  bool recursive{true};
  ProcessorConfig processor_config;
};

struct BatchResult
{
  size_t total_bags_processed{0};
  size_t total_overrides_found{0};
  size_t failed_bags{0};
  std::vector<ProcessResult> results;
};

}  // namespace override_event_extractor

#endif  // TYPE_ALIAS_HPP_
