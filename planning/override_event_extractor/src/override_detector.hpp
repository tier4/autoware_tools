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

#ifndef OVERRIDE_DETECTOR_HPP_
#define OVERRIDE_DETECTOR_HPP_

#include "type_alias.hpp"

#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace override_event_extractor
{

class OverrideDetector
{
public:
  explicit OverrideDetector(const DetectorConfig & config);

  std::vector<OverrideEvent> detect(const std::string & bag_path);

private:
  bool isOverrideActive(uint8_t mode) const;

  void applyMargins(OverrideEvent & event) const;

  std::vector<OverrideEvent> mergeOverlapping(const std::vector<OverrideEvent> & events) const;

  bool meetsMinimumDuration(const TimeRange & range) const;

  DetectorConfig config_;
};

}  // namespace override_event_extractor

#endif  // OVERRIDE_DETECTOR_HPP_
