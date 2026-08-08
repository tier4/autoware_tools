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

#ifndef MPPI_EVALUATION_TYPES_HPP_
#define MPPI_EVALUATION_TYPES_HPP_

#include "autoware/mppi_optimizer/first_order_dubins_mppi_interface.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace autoware::mppi_optimizer::evaluation
{

struct MppiInputFrame
{
  std::string frame_id;
  uint64_t timestamp_ns{0};
  Trajectory reference_trajectory;
  Odometry odometry;
  std::optional<double> initial_steer_rad;
  std::optional<double> initial_accel_mps2;
  TrackedObjects tracked_objects;
  std::vector<Segment> road_borders;
};

struct MppiConfiguration
{
  std::string name;
  FirstOrderDubinsMppiCostParams cost_params;
  FirstOrderDubinsMppiRuntimeOptions runtime_options;
  // Extend here if you add horizon/sampling parameters to your planner
};

struct FrameEvaluationMetrics
{
  double execution_time_ms{0.0};
  double baseline_cost{0.0};
  bool was_rejected{false};
  bool is_valid{false};
  std::optional<std::size_t> first_invalid_index;

  // Custom Kinematic & Safety Metrics
  double max_lateral_acceleration_mps2{0.0};
  double max_jerk_mps3{0.0};
  double max_front_wheel_angle_rate_rps{0.0};
  double max_cross_track_error_m{0.0};
  double min_obstacle_clearance_m{1e9};
  double min_road_border_clearance_m{1e9};
};

struct EvaluatedFrameResult
{
  std::string frame_id;
  std::string config_name;
  FirstOrderDubinsMppiOptimizationResult optimize_result;
  FrameEvaluationMetrics metrics;
};

}  // namespace autoware::mppi_optimizer::evaluation

#endif  // MPPI_EVALUATION_TYPES_HPP_
