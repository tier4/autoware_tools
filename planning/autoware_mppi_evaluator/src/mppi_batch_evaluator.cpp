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

#include "autoware/mppi_evaluator/mppi_batch_evaluator.hpp"

#include "autoware/avoidance_target_detector/boundary.hpp"
#include "autoware/avoidance_target_detector/object_filtering.hpp"
#include "autoware/mppi_optimizer/predicted_objects_obstacles.hpp"
#include "autoware/mppi_optimizer/tracked_objects_obstacles.hpp"

#include <rclcpp/time.hpp>

#include <boost/geometry.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autoware::mppi_evaluator
{
namespace
{

constexpr double kMppiDt = 0.1;

std::vector<mppi_optimizer::Segment> to_mppi_segments(
  const std::vector<avoidance_target_detector::Segment> & segments)
{
  std::vector<mppi_optimizer::Segment> result;
  result.reserve(segments.size());
  for (const auto & segment : segments) {
    result.push_back(
      {static_cast<float>(boost::geometry::get<0, 0>(segment)),
       static_cast<float>(boost::geometry::get<0, 1>(segment)),
       static_cast<float>(boost::geometry::get<1, 0>(segment)),
       static_cast<float>(boost::geometry::get<1, 1>(segment))});
  }
  return result;
}

double point_to_segment_distance(
  const double px, const double py, const mppi_optimizer::Segment & segment)
{
  const double vx = static_cast<double>(segment.x1 - segment.x0);
  const double vy = static_cast<double>(segment.y1 - segment.y0);
  const double length_squared = vx * vx + vy * vy;
  if (length_squared <= std::numeric_limits<double>::epsilon()) {
    return std::hypot(px - segment.x0, py - segment.y0);
  }
  const double projection =
    std::clamp(((px - segment.x0) * vx + (py - segment.y0) * vy) / length_squared, 0.0, 1.0);
  return std::hypot(px - (segment.x0 + projection * vx), py - (segment.y0 + projection * vy));
}

double point_to_trajectory_distance(
  const double x, const double y, const mppi_optimizer::Trajectory & trajectory)
{
  if (trajectory.points.empty()) {
    return 0.0;
  }
  if (trajectory.points.size() == 1U) {
    const auto & point = trajectory.points.front().pose.position;
    return std::hypot(x - point.x, y - point.y);
  }
  double distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 1U; index < trajectory.points.size(); ++index) {
    const auto & first = trajectory.points[index - 1U].pose.position;
    const auto & second = trajectory.points[index].pose.position;
    const mppi_optimizer::Segment segment{
      static_cast<float>(first.x), static_cast<float>(first.y), static_cast<float>(second.x),
      static_cast<float>(second.y)};
    distance = std::min(distance, point_to_segment_distance(x, y, segment));
  }
  return distance;
}

double point_to_segments_distance(
  const double x, const double y, const std::vector<mppi_optimizer::Segment> & segments)
{
  double distance = std::numeric_limits<double>::infinity();
  for (const auto & segment : segments) {
    distance = std::min(distance, point_to_segment_distance(x, y, segment));
  }
  return distance;
}

double obstacle_clearance(
  const mppi_optimizer::Trajectory & trajectory,
  const mppi_optimizer::TrackedObjects & tracked_objects,
  const mppi_optimizer::FirstOrderDubinsMppiVehicleParams & vehicle_params)
{
  if (trajectory.points.empty() || tracked_objects.objects.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  constexpr int kHorizonSteps = 80;
  std::vector<float> obstacle_x;
  std::vector<float> obstacle_y;
  std::vector<float> obstacle_yaw;
  std::vector<float> obstacle_half_length;
  std::vector<float> obstacle_half_width;
  mppi_optimizer::buildObstacleTrajectoryBuffersFromTrackedObjects(
    tracked_objects, static_cast<float>(kMppiDt), kHorizonSteps, obstacle_x, obstacle_y,
    obstacle_yaw, obstacle_half_length, obstacle_half_width);

  const double ego_radius = 0.5 * std::hypot(vehicle_params.ego_length, vehicle_params.ego_width);
  double clearance = std::numeric_limits<double>::infinity();
  const std::size_t obstacle_count = obstacle_half_length.size();
  const std::size_t step_count = std::min<std::size_t>(trajectory.points.size(), kHorizonSteps);
  for (std::size_t obstacle_index = 0U; obstacle_index < obstacle_count; ++obstacle_index) {
    const double obstacle_radius =
      std::hypot(obstacle_half_length[obstacle_index], obstacle_half_width[obstacle_index]);
    for (std::size_t step = 0U; step < step_count; ++step) {
      const auto & ego_pose = trajectory.points[step].pose;
      const double ego_yaw = tf2::getYaw(ego_pose.orientation);
      const double ego_x =
        ego_pose.position.x + vehicle_params.ego_axle_to_box_center * std::cos(ego_yaw);
      const double ego_y =
        ego_pose.position.y + vehicle_params.ego_axle_to_box_center * std::sin(ego_yaw);
      const std::size_t buffer_index = obstacle_index * kHorizonSteps + step;
      const double center_distance =
        std::hypot(ego_x - obstacle_x[buffer_index], ego_y - obstacle_y[buffer_index]);
      clearance =
        std::min(clearance, std::max(0.0, center_distance - ego_radius - obstacle_radius));
    }
  }
  return clearance;
}

void compute_metrics(
  const MppiInputFrame & frame, const MppiConfiguration & configuration,
  EvaluatedFrameResult & evaluated)
{
  const auto & debug = evaluated.optimize_result.debug;
  auto & metrics = evaluated.metrics;
  metrics.baseline_cost = debug.baseline_cost;
  metrics.was_rejected = debug.was_rejected;
  metrics.is_valid = debug.validation.isValid();
  metrics.first_invalid_index = debug.validation.first_invalid_index;
  metrics.invalidity_reasons = static_cast<std::uint8_t>(debug.validation.reasons);
  metrics.min_effective_sample_size = debug.iteration_diagnostics.min_ess;
  metrics.max_importance_weight = debug.iteration_diagnostics.max_weight;
  metrics.selected_object_count = evaluated.selected_objects.objects.size();
  metrics.road_border_segment_count = evaluated.road_borders.size();
  metrics.drivable_area_segment_count = evaluated.drivable_area.size();

  const auto & points = debug.optimized_trajectory.points;
  const double ego_radius =
    0.5 *
    std::hypot(configuration.vehicle_params.ego_length, configuration.vehicle_params.ego_width);
  double cross_track_error_sum = 0.0;
  for (const auto & point : points) {
    const double error = point_to_trajectory_distance(
      point.pose.position.x, point.pose.position.y, frame.reference_trajectory);
    metrics.max_cross_track_error_m = std::max(metrics.max_cross_track_error_m, error);
    cross_track_error_sum += error;

    const double speed = std::abs(static_cast<double>(point.longitudinal_velocity_mps));
    const double steer = static_cast<double>(point.front_wheel_angle_rad);
    const double lateral_acceleration =
      speed * speed * std::tan(steer) /
      std::max(1.0e-6, static_cast<double>(configuration.vehicle_params.wheel_base));
    metrics.max_lateral_acceleration_mps2 =
      std::max(metrics.max_lateral_acceleration_mps2, std::abs(lateral_acceleration));

    const double yaw = tf2::getYaw(point.pose.orientation);
    const double center_x =
      point.pose.position.x + configuration.vehicle_params.ego_axle_to_box_center * std::cos(yaw);
    const double center_y =
      point.pose.position.y + configuration.vehicle_params.ego_axle_to_box_center * std::sin(yaw);
    const double road_distance =
      point_to_segments_distance(center_x, center_y, evaluated.road_borders);
    const double drivable_distance =
      point_to_segments_distance(center_x, center_y, evaluated.drivable_area);
    metrics.min_road_border_clearance_m =
      std::min(metrics.min_road_border_clearance_m, std::max(0.0, road_distance - ego_radius));
    metrics.min_drivable_area_clearance_m = std::min(
      metrics.min_drivable_area_clearance_m, std::max(0.0, drivable_distance - ego_radius));
  }
  if (!points.empty()) {
    metrics.mean_cross_track_error_m = cross_track_error_sum / points.size();
  }
  for (std::size_t index = 1U; index < points.size(); ++index) {
    metrics.max_acceleration_command_rate_mps3 = std::max(
      metrics.max_acceleration_command_rate_mps3,
      std::abs(
        static_cast<double>(points[index].acceleration_mps2) -
        static_cast<double>(points[index - 1U].acceleration_mps2)) /
        kMppiDt);
    metrics.max_front_wheel_angle_rate_rps = std::max(
      metrics.max_front_wheel_angle_rate_rps,
      std::abs(
        static_cast<double>(points[index].front_wheel_angle_rad) -
        static_cast<double>(points[index - 1U].front_wheel_angle_rad)) /
        kMppiDt);
  }
  metrics.min_obstacle_clearance_m = obstacle_clearance(
    debug.optimized_trajectory, evaluated.selected_objects, configuration.vehicle_params);
}

}  // namespace

struct MppiEvaluationSession::Impl
{
  Impl(MppiEnvironment environment_in, MppiConfiguration configuration_in, EvaluationMode mode_in)
  : environment(std::move(environment_in)),
    configuration(std::move(configuration_in)),
    mode(mode_in)
  {
    route_handler = std::make_unique<avoidance_target_detector::ExtendedRouteHandler>(
      environment.lanelet_map, environment.route);
    route_handler->create_map();
    reset_optimizer();
  }

  void reset_optimizer()
  {
    optimizer = std::make_unique<mppi_optimizer::FirstOrderDubinsMppiInterface>();
    optimizer->setVehicleParams(configuration.vehicle_params);
    optimizer->setCostParams(configuration.cost_params);
    optimizer->setRuntimeOptions(configuration.runtime_options);
  }

  void reset()
  {
    reset_optimizer();
    object_selector = avoidance_target_detector::TrackedObjectSelector{};
    last_timestamp_ns.reset();
  }

  mppi_optimizer::TrackedObjects select_objects(const MppiInputFrame & frame)
  {
    const double half_length = 0.5 * static_cast<double>(configuration.vehicle_params.ego_length);
    const double half_width = 0.5 * static_cast<double>(configuration.vehicle_params.ego_width);
    const double max_longitudinal_offset =
      std::abs(static_cast<double>(configuration.vehicle_params.ego_axle_to_box_center)) +
      half_length;
    const double margin = std::hypot(max_longitudinal_offset, half_width) +
                          configuration.cost_params.boundary_threshold;
    const auto objects_in_range = avoidance_target_detector::filter_objects_in_range(
      frame.tracked_objects, frame.reference_trajectory, margin);

    if (mode == EvaluationMode::isolated) {
      return objects_in_range;
    }

    const auto current_time = rclcpp::Time(static_cast<std::int64_t>(frame.timestamp_ns));
    object_selector.update_objects(
      current_time, objects_in_range, frame.reference_trajectory, *route_handler);
    auto avoidance_targets = object_selector.get_avoidance_targets(
      objects_in_range, frame.reference_trajectory, route_handler->get_extended_route_bounds());
    const auto driving_along_targets = object_selector.get_driving_along_vehicles(objects_in_range);
    avoidance_targets.objects.insert(
      avoidance_targets.objects.end(), driving_along_targets.objects.begin(),
      driving_along_targets.objects.end());
    return avoidance_targets;
  }

  EvaluatedFrameResult evaluate(const MppiInputFrame & frame)
  {
    if (frame.reference_trajectory.points.empty()) {
      throw std::invalid_argument("The reference trajectory is empty");
    }
    if (
      mode == EvaluationMode::chronological && last_timestamp_ns.has_value() &&
      frame.timestamp_ns <= *last_timestamp_ns) {
      throw std::invalid_argument("Chronological frames must use strictly increasing timestamps");
    }
    if (mode == EvaluationMode::isolated) {
      reset_optimizer();
    }

    EvaluatedFrameResult evaluated;
    evaluated.frame_id = frame.frame_id;
    evaluated.config_name = configuration.name;
    evaluated.timestamp_ns = frame.timestamp_ns;
    evaluated.selected_objects = select_objects(frame);

    const double half_length = 0.5 * static_cast<double>(configuration.vehicle_params.ego_length);
    const double axle_offset =
      std::abs(static_cast<double>(configuration.vehicle_params.ego_axle_to_box_center));
    const double margin = half_length + axle_offset + configuration.boundary_search_margin_m;
    evaluated.road_borders = to_mppi_segments(
      route_handler->get_road_borders_around_trajectory(frame.reference_trajectory, margin));
    evaluated.drivable_area = to_mppi_segments(
      route_handler->get_drivable_area_around_trajectory(frame.reference_trajectory, margin));

    const auto start = std::chrono::steady_clock::now();
    evaluated.optimize_result = optimizer->optimizeTrajectory(
      frame.reference_trajectory, frame.odometry, frame.acceleration, frame.steering_status,
      evaluated.selected_objects, evaluated.road_borders, evaluated.drivable_area);
    const auto end = std::chrono::steady_clock::now();
    evaluated.metrics.execution_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
    compute_metrics(frame, configuration, evaluated);
    last_timestamp_ns = frame.timestamp_ns;
    return evaluated;
  }

  MppiEnvironment environment;
  MppiConfiguration configuration;
  EvaluationMode mode;
  std::unique_ptr<avoidance_target_detector::ExtendedRouteHandler> route_handler;
  avoidance_target_detector::TrackedObjectSelector object_selector;
  std::unique_ptr<mppi_optimizer::FirstOrderDubinsMppiInterface> optimizer;
  std::optional<std::uint64_t> last_timestamp_ns;
};

MppiEvaluationSession::MppiEvaluationSession(
  MppiEnvironment environment, MppiConfiguration configuration, const EvaluationMode mode)
: impl_(std::make_unique<Impl>(std::move(environment), std::move(configuration), mode))
{
}

MppiEvaluationSession::~MppiEvaluationSession() = default;
MppiEvaluationSession::MppiEvaluationSession(MppiEvaluationSession &&) noexcept = default;
MppiEvaluationSession & MppiEvaluationSession::operator=(MppiEvaluationSession &&) noexcept =
  default;

EvaluatedFrameResult MppiEvaluationSession::evaluate(const MppiInputFrame & frame)
{
  return impl_->evaluate(frame);
}

void MppiEvaluationSession::reset()
{
  impl_->reset();
}

MppiBatchEvaluator::MppiBatchEvaluator(MppiEnvironment environment)
: environment_(std::move(environment))
{
}

std::vector<EvaluatedFrameResult> MppiBatchEvaluator::evaluate(
  const std::vector<MppiInputFrame> & frames, const std::vector<MppiConfiguration> & configurations,
  const EvaluationMode mode) const
{
  std::vector<EvaluatedFrameResult> results;
  results.reserve(frames.size() * configurations.size());
  for (const auto & configuration : configurations) {
    MppiEvaluationSession session(environment_, configuration, mode);
    for (const auto & frame : frames) {
      results.push_back(session.evaluate(frame));
    }
  }
  return results;
}

}  // namespace autoware::mppi_evaluator
