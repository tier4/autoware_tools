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

#ifndef BAG_PROCESSOR_HPP_
#define BAG_PROCESSOR_HPP_

#include "bag_splitter.hpp"
#include "override_detector.hpp"
#include "type_alias.hpp"

#include <string>

namespace override_event_extractor
{

class BagProcessor
{
public:
  explicit BagProcessor(const ProcessorConfig & config);

  ProcessResult process(
    const std::string & bag_path, const std::string & output_dir,
    const RouteMessage & route_msg = RouteMessage());

private:
  void generateSummary(const ProcessResult & result, const std::string & output_dir) const;

  std::string getBagBaseName(const std::string & bag_path) const;

  ProcessorConfig config_;
  OverrideDetector detector_;
  BagSplitter splitter_;
};

}  // namespace override_event_extractor

#endif  // BAG_PROCESSOR_HPP_
