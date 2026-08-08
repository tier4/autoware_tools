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

#ifndef MPPI_BATCH_EVALUATOR_HPP_
#define MPPI_BATCH_EVALUATOR_HPP_

#include "autoware/mppi_optimizer/evaluation/mppi_evaluation_types.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace autoware::mppi_optimizer::evaluation
{

class MppiBatchEvaluator
{
public:
  MppiBatchEvaluator() = default;

  std::vector<EvaluatedFrameResult> evaluate(
    const std::vector<MppiInputFrame> & frames, const std::vector<MppiConfiguration> & configs)
  {
    std::vector<EvaluatedFrameResult> results;
    results.reserve(frames.size() * configs.size());

    // Reuse a single interface across frames for a given config to avoid CUDA reallocation
    auto interface = std::make_unique<FirstOrderDubinsMppiInterface>();

    for (const auto & config : configs) {
      interface->setCostParams(config.cost_params);
      interface->setRuntimeOptions(config.runtime_options);

      for (const auto & frame : frames) {
        EvaluatedFrameResult res;
        res.frame_id = frame.frame_id;
        res.config_name = config.name;

        // Measure execution latency
        const auto start_time = std::chrono::high_resolution_clock::now();
        res.optimize_result = interface->optimizeTrajectory(
          frame.reference_trajectory, frame.odometry, frame.initial_steer_rad,
          frame.initial_accel_mps2, frame.tracked_objects, frame.road_borders, {});
        const auto end_time = std::chrono::high_resolution_clock::now();

        res.metrics.execution_time_ms =
          std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // Calculate quantitative metrics
        computeMetrics(frame, res.optimize_result, res.metrics);
        results.push_back(std::move(res));
      }
    }
    return results;
  }

private:
  void computeMetrics(
    const MppiInputFrame & frame, const FirstOrderDubinsMppiOptimizationResult & result,
    FrameEvaluationMetrics & metrics)
  {
    metrics.baseline_cost = result.debug.baseline_cost;
    metrics.was_rejected = result.debug.was_rejected;
    metrics.is_valid = result.debug.validation.isValid();
    metrics.first_invalid_index = result.debug.validation.first_invalid_index;

    const auto & points = result.trajectory.points;
    if (points.size() < 2) {
      return;
    }

    // Example: Smoothness & tracking error computation along the horizon
    for (std::size_t i = 1; i < points.size(); ++i) {
      const double dt = std::max(0.01, getDeltaTime(points[i - 1], points[i]));
      const double jerk =
        std::abs(points[i].acceleration_mps2 - points[i - 1].acceleration_mps2) / dt;
      metrics.max_jerk_mps3 = std::max(metrics.max_jerk_mps3, jerk);

      const double steer_rate =
        std::abs(points[i].front_wheel_angle_rad - points[i - 1].front_wheel_angle_rad) / dt;
      metrics.max_front_wheel_angle_rate_rps =
        std::max(metrics.max_front_wheel_angle_rate_rps, steer_rate);
    }

    // (Additional geometry checks for distance-to-border/obstacle can be placed here)
  }

  static double getDeltaTime(
    const autoware_planning_msgs::msg::TrajectoryPoint & p1,
    const autoware_planning_msgs::msg::TrajectoryPoint & p2)
  {
    const double t1 = p1.time_from_start.sec + p1.time_from_start.nanosec * 1e-9;
    const double t2 = p2.time_from_start.sec + p2.time_from_start.nanosec * 1e-9;
    return t2 - t1;
  }
};

}  // namespace autoware::mppi_optimizer::evaluation

#endif  // MPPI_BATCH_EVALUATOR_HPP_
