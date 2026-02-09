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

#include "override_detector.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

namespace override_event_extractor
{

OverrideDetector::OverrideDetector(const DetectorConfig & config) : config_(config) {}

std::vector<OverrideEvent> OverrideDetector::detect(const std::string & bag_path)
{
  rosbag2_cpp::Reader reader;
  reader.open(bag_path);

  rosbag2_storage::StorageFilter filter;
  filter.topics.push_back(config_.control_mode_topic);
  reader.set_filter(filter);

  std::vector<OverrideEvent> raw_events;
  std::optional<int64_t> override_start = std::nullopt;
  uint8_t last_mode = config_.autonomous_mode_value;

  rclcpp::Serialization<ControlModeReport> serializer;

  while (reader.has_next()) {
    auto bag_message = reader.read_next();

    if (bag_message->topic_name != config_.control_mode_topic) {
      continue;
    }

    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
    ControlModeReport msg;
    serializer.deserialize_message(&serialized_msg, &msg);

    const auto current_mode = msg.mode;
    const auto timestamp = bag_message->time_stamp;

    if (!isOverrideActive(last_mode) && isOverrideActive(current_mode)) {
      override_start = timestamp;
    } else if (isOverrideActive(last_mode) && !isOverrideActive(current_mode)) {
      if (override_start.has_value()) {
        TimeRange raw_range{override_start.value(), timestamp};

        if (!config_.filter_brief_overrides || meetsMinimumDuration(raw_range)) {
          OverrideEvent event(raw_range, raw_range, raw_events.size());
          applyMargins(event);
          raw_events.push_back(event);
        }

        override_start.reset();
      }
    }

    last_mode = current_mode;
  }

  if (override_start.has_value()) {
    auto metadata = reader.get_metadata();
    auto end_time = duration_cast<nanoseconds>(metadata.duration).count() +
                    duration_cast<nanoseconds>(metadata.starting_time.time_since_epoch()).count();

    TimeRange raw_range{override_start.value(), end_time};
    if (!config_.filter_brief_overrides || meetsMinimumDuration(raw_range)) {
      OverrideEvent event(raw_range, raw_range, raw_events.size());
      applyMargins(event);
      raw_events.push_back(event);
    }
  }

  return mergeOverlapping(raw_events);
}

bool OverrideDetector::isOverrideActive(uint8_t mode) const
{
  return mode == config_.override_mode_value;
}

void OverrideDetector::applyMargins(OverrideEvent & event) const
{
  const auto pre_margin_ns = static_cast<int64_t>(config_.pre_margin_sec * 1e9);
  const auto post_margin_ns = static_cast<int64_t>(config_.post_margin_sec * 1e9);

  const int64_t raw_start = event.raw_range.start_ns;

  event.extended_range.start_ns = std::max<int64_t>(0, raw_start - pre_margin_ns);
  event.extended_range.end_ns = raw_start + post_margin_ns;
}

std::vector<OverrideEvent> OverrideDetector::mergeOverlapping(
  const std::vector<OverrideEvent> & events) const
{
  if (events.empty()) {
    return {};
  }

  std::vector<OverrideEvent> sorted_events = events;
  std::sort(sorted_events.begin(), sorted_events.end(), [](const auto & a, const auto & b) {
    return a.extended_range.start_ns < b.extended_range.start_ns;
  });

  std::vector<OverrideEvent> merged;
  merged.push_back(sorted_events[0]);

  for (size_t i = 1; i < sorted_events.size(); ++i) {
    auto & last = merged.back();
    const auto & current = sorted_events[i];

    if (last.extended_range.overlaps(current.extended_range)) {
      last.extended_range = last.extended_range.merge(current.extended_range);
      last.raw_range = last.raw_range.merge(current.raw_range);
    } else {
      merged.push_back(current);
    }
  }

  for (size_t i = 0; i < merged.size(); ++i) {
    merged[i].index = i;
  }

  return merged;
}

bool OverrideDetector::meetsMinimumDuration(const TimeRange & range) const
{
  return range.duration_seconds() >= config_.min_override_duration_sec;
}

}  // namespace override_event_extractor
