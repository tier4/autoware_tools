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

#include "autoware/mppi_evaluator/mppi_evaluation_types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace autoware::mppi_evaluator
{

TEST(MppiEvaluationTypes, UsesSafeMetricDefaults)
{
  const FrameEvaluationMetrics metrics;
  EXPECT_FALSE(metrics.was_rejected);
  EXPECT_FALSE(metrics.is_valid);
  EXPECT_FALSE(metrics.first_invalid_index.has_value());
  EXPECT_EQ(metrics.invalidity_reasons, std::uint8_t{0U});
  EXPECT_TRUE(std::isinf(metrics.min_obstacle_clearance_m));
  EXPECT_TRUE(std::isinf(metrics.min_road_border_clearance_m));
  EXPECT_TRUE(std::isinf(metrics.min_drivable_area_clearance_m));
}

TEST(MppiEvaluationTypes, UsesIsolatedAndChronologicalModes)
{
  EXPECT_NE(EvaluationMode::isolated, EvaluationMode::chronological);
}

}  // namespace autoware::mppi_evaluator
