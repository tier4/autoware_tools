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

#ifndef AUTOWARE__MPPI_EVALUATOR__MPPI_EVALUATION_TYPES_HPP_
#define AUTOWARE__MPPI_EVALUATOR__MPPI_EVALUATION_TYPES_HPP_

#include "autoware/mppi_optimizer/first_order_dubins_mppi_interface.hpp"

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace autoware::mppi_evaluator
{

enum class EvaluationMode { isolated, chronological };

struct MppiInputFrame
{
  std::string frame_id;
  std::uint64_t timestamp_ns{0U};
  mppi_optimizer::Trajectory reference_trajectory;
  mppi_optimizer::Odometry odometry;
  std::optional<geometry_msgs::msg::AccelWithCovarianceStamped> acceleration;
  std::optional<autoware_vehicle_msgs::msg::SteeringReport> steering_status;
  mppi_optimizer::TrackedObjects tracked_objects;
  double odometry_age_ms{0.0};
  std::optional<double> acceleration_age_ms;
  std::optional<double> steering_age_ms;
  std::optional<double> tracked_objects_age_ms;
};

struct MppiEnvironment
{
  autoware_map_msgs::msg::LaneletMapBin lanelet_map;
  autoware_planning_msgs::msg::LaneletRoute route;
};

struct MppiConfiguration
{
  std::string name{"default"};
  mppi_optimizer::FirstOrderDubinsMppiCostParams cost_params;
  mppi_optimizer::FirstOrderDubinsMppiRuntimeOptions runtime_options;
  mppi_optimizer::FirstOrderDubinsMppiVehicleParams vehicle_params;
  double boundary_search_margin_m{1.0};
};

struct FrameEvaluationMetrics
{
  double execution_time_ms{0.0};
  double baseline_cost{0.0};
  bool was_rejected{false};
  bool is_valid{false};
  std::optional<std::size_t> first_invalid_index;
  std::uint8_t invalidity_reasons{0U};
  double min_effective_sample_size{0.0};
  double max_importance_weight{0.0};
  double max_lateral_acceleration_mps2{0.0};
  double max_acceleration_command_rate_mps3{0.0};
  double max_front_wheel_angle_rate_rps{0.0};
  double max_cross_track_error_m{0.0};
  double mean_cross_track_error_m{0.0};
  double min_obstacle_clearance_m{std::numeric_limits<double>::infinity()};
  double min_road_border_clearance_m{std::numeric_limits<double>::infinity()};
  double min_drivable_area_clearance_m{std::numeric_limits<double>::infinity()};
  std::size_t selected_object_count{0U};
  std::size_t road_border_segment_count{0U};
  std::size_t drivable_area_segment_count{0U};
};

struct EvaluatedFrameResult
{
  std::string frame_id;
  std::string config_name;
  std::uint64_t timestamp_ns{0U};
  mppi_optimizer::FirstOrderDubinsMppiOptimizationResult optimize_result;
  FrameEvaluationMetrics metrics;
  mppi_optimizer::TrackedObjects selected_objects;
  std::vector<mppi_optimizer::Segment> road_borders;
  std::vector<mppi_optimizer::Segment> drivable_area;
};

}  // namespace autoware::mppi_evaluator

#endif  // AUTOWARE__MPPI_EVALUATOR__MPPI_EVALUATION_TYPES_HPP_
