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

#include "autoware/avoidance_target_detector/object_filtering.hpp"
#include "autoware/mppi_evaluator/mppi_batch_evaluator.hpp"
#include "autoware/mppi_optimizer/predicted_objects_obstacles.hpp"

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tf2/utils.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace autoware::mppi_evaluator
{
namespace
{

template <class Message>
Message deserialize_message(const py::bytes & value)
{
  const std::string bytes = value;
  if (bytes.empty()) {
    throw std::invalid_argument("A serialized ROS message is empty");
  }
  rclcpp::SerializedMessage serialized(bytes.size());
  auto & raw = serialized.get_rcl_serialized_message();
  std::memcpy(raw.buffer, bytes.data(), bytes.size());
  raw.buffer_length = bytes.size();
  Message message;
  rclcpp::Serialization<Message> serializer;
  serializer.deserialize_message(&serialized, &message);
  return message;
}

template <class Message>
std::optional<Message> deserialize_optional_message(const py::object & value)
{
  if (value.is_none()) {
    return std::nullopt;
  }
  return deserialize_message<Message>(value.cast<py::bytes>());
}

py::dict trajectory_to_dict(const mppi_optimizer::Trajectory & trajectory)
{
  py::list points;
  for (const auto & point : trajectory.points) {
    py::dict item;
    item["x"] = point.pose.position.x;
    item["y"] = point.pose.position.y;
    item["z"] = point.pose.position.z;
    item["yaw"] = tf2::getYaw(point.pose.orientation);
    item["velocity_mps"] = point.longitudinal_velocity_mps;
    item["acceleration_mps2"] = point.acceleration_mps2;
    item["front_wheel_angle_rad"] = point.front_wheel_angle_rad;
    item["time_from_start_ns"] =
      static_cast<std::int64_t>(point.time_from_start.sec) * 1000000000LL +
      static_cast<std::int64_t>(point.time_from_start.nanosec);
    points.append(std::move(item));
  }
  py::dict result;
  result["frame_id"] = trajectory.header.frame_id;
  result["points"] = std::move(points);
  return result;
}

py::list segments_to_list(const std::vector<mppi_optimizer::Segment> & segments)
{
  py::list result;
  for (const auto & segment : segments) {
    result.append(py::make_tuple(segment.x0, segment.y0, segment.x1, segment.y1));
  }
  return result;
}

py::list objects_to_list(const mppi_optimizer::TrackedObjects & objects)
{
  py::list result;
  for (const auto & object : objects.objects) {
    float length = mppi_optimizer::kDefaultObstacleLength;
    float width = mppi_optimizer::kDefaultObstacleWidth;
    mppi_optimizer::obstacleFootprintFromShape(object.shape, length, width);
    const auto & pose = object.kinematics.pose_with_covariance.pose;
    py::dict item;
    item["x"] = pose.position.x;
    item["y"] = pose.position.y;
    item["yaw"] = tf2::getYaw(pose.orientation);
    item["length"] = length;
    item["width"] = width;
    result.append(std::move(item));
  }
  return result;
}

py::dict metrics_to_dict(const FrameEvaluationMetrics & metrics)
{
  py::dict result;
  result["execution_time_ms"] = metrics.execution_time_ms;
  result["baseline_cost"] = metrics.baseline_cost;
  result["was_rejected"] = metrics.was_rejected;
  result["is_valid"] = metrics.is_valid;
  if (metrics.first_invalid_index.has_value()) {
    result["first_invalid_index"] = *metrics.first_invalid_index;
  } else {
    result["first_invalid_index"] = py::none();
  }
  result["invalidity_reasons"] = metrics.invalidity_reasons;
  result["min_effective_sample_size"] = metrics.min_effective_sample_size;
  result["max_importance_weight"] = metrics.max_importance_weight;
  result["max_lateral_acceleration_mps2"] = metrics.max_lateral_acceleration_mps2;
  result["max_acceleration_command_rate_mps3"] = metrics.max_acceleration_command_rate_mps3;
  result["max_front_wheel_angle_rate_rps"] = metrics.max_front_wheel_angle_rate_rps;
  result["max_cross_track_error_m"] = metrics.max_cross_track_error_m;
  result["mean_cross_track_error_m"] = metrics.mean_cross_track_error_m;
  result["min_obstacle_clearance_m"] = metrics.min_obstacle_clearance_m;
  result["min_road_border_clearance_m"] = metrics.min_road_border_clearance_m;
  result["min_drivable_area_clearance_m"] = metrics.min_drivable_area_clearance_m;
  result["selected_object_count"] = metrics.selected_object_count;
  result["road_border_segment_count"] = metrics.road_border_segment_count;
  result["drivable_area_segment_count"] = metrics.drivable_area_segment_count;
  return result;
}

py::dict result_to_dict(const EvaluatedFrameResult & evaluated)
{
  py::dict result;
  result["frame_id"] = evaluated.frame_id;
  result["config_name"] = evaluated.config_name;
  result["timestamp_ns"] = evaluated.timestamp_ns;
  result["trajectory"] = trajectory_to_dict(evaluated.optimize_result.trajectory);
  result["reference_trajectory"] =
    trajectory_to_dict(evaluated.optimize_result.debug.reference_trajectory);
  result["optimized_trajectory"] =
    trajectory_to_dict(evaluated.optimize_result.debug.optimized_trajectory);
  result["metrics"] = metrics_to_dict(evaluated.metrics);
  result["road_borders"] = segments_to_list(evaluated.road_borders);
  result["drivable_area"] = segments_to_list(evaluated.drivable_area);
  result["selected_objects"] = objects_to_list(evaluated.selected_objects);
  result["ess_per_iteration"] =
    evaluated.optimize_result.debug.iteration_diagnostics.ess_per_iteration;
  result["max_weight_per_iteration"] =
    evaluated.optimize_result.debug.iteration_diagnostics.max_weight_per_iteration;
  return result;
}

EvaluationMode parse_mode(const std::string & mode)
{
  if (mode == "isolated") {
    return EvaluationMode::isolated;
  }
  if (mode == "chronological") {
    return EvaluationMode::chronological;
  }
  throw std::invalid_argument("Evaluation mode must be 'isolated' or 'chronological'");
}

}  // namespace
}  // namespace autoware::mppi_evaluator

PYBIND11_MODULE(mppi_optimizer_py, module)
{
  using autoware::mppi_evaluator::deserialize_message;
  using autoware::mppi_evaluator::deserialize_optional_message;
  using autoware::mppi_evaluator::EvaluatedFrameResult;
  using autoware::mppi_evaluator::MppiConfiguration;
  using autoware::mppi_evaluator::MppiEnvironment;
  using autoware::mppi_evaluator::MppiEvaluationSession;
  using autoware::mppi_evaluator::MppiInputFrame;
  using autoware::mppi_evaluator::objects_to_list;
  using autoware::mppi_evaluator::parse_mode;
  using autoware::mppi_evaluator::result_to_dict;
  using autoware::mppi_evaluator::trajectory_to_dict;
  using autoware::mppi_optimizer::FirstOrderDubinsMppiCostParams;
  using autoware::mppi_optimizer::FirstOrderDubinsMppiRuntimeOptions;
  using autoware::mppi_optimizer::FirstOrderDubinsMppiVehicleParams;

  module.doc() = "Native bindings for offline Autoware MPPI evaluation";

  module.def(
    "deserialize_trajectory",
    [](const py::bytes & cdr_bytes) {
      return trajectory_to_dict(
        deserialize_message<autoware_planning_msgs::msg::Trajectory>(cdr_bytes));
    },
    py::arg("cdr_bytes"));

  module.def(
    "deserialize_tracked_objects",
    [](const py::bytes & cdr_bytes) {
      return objects_to_list(
        deserialize_message<autoware_perception_msgs::msg::TrackedObjects>(cdr_bytes));
    },
    py::arg("cdr_bytes"));

  module.def(
    "deserialize_tracked_objects_in_range",
    [](const py::bytes & objects_cdr, const py::bytes & trajectory_cdr, const double margin) {
      const auto objects =
        deserialize_message<autoware_perception_msgs::msg::TrackedObjects>(objects_cdr);
      const auto trajectory =
        deserialize_message<autoware_planning_msgs::msg::Trajectory>(trajectory_cdr);
      return objects_to_list(
        autoware::avoidance_target_detector::filter_objects_in_range(objects, trajectory, margin));
    },
    py::arg("objects_cdr"), py::arg("trajectory_cdr"), py::arg("margin"));

  py::class_<FirstOrderDubinsMppiCostParams>(module, "CostParams")
    .def(py::init<>())
    .def_readwrite("lambda_", &FirstOrderDubinsMppiCostParams::lambda)
    .def_readwrite("speed_coeff", &FirstOrderDubinsMppiCostParams::speed_coeff)
    .def_readwrite("track_coeff", &FirstOrderDubinsMppiCostParams::track_coeff)
    .def_readwrite("track_terminal_scale", &FirstOrderDubinsMppiCostParams::track_terminal_scale)
    .def_readwrite("heading_coeff", &FirstOrderDubinsMppiCostParams::heading_coeff)
    .def_readwrite(
      "lateral_distance_coeff", &FirstOrderDubinsMppiCostParams::lateral_distance_coeff)
    .def_readwrite(
      "lateral_yaw_error_coeff", &FirstOrderDubinsMppiCostParams::lateral_yaw_error_coeff)
    .def_readwrite("crash_coeff", &FirstOrderDubinsMppiCostParams::crash_coeff)
    .def_readwrite("boundary_threshold", &FirstOrderDubinsMppiCostParams::boundary_threshold)
    .def_readwrite("accel_cmd_coeff", &FirstOrderDubinsMppiCostParams::accel_cmd_coeff)
    .def_readwrite("steer_cmd_coeff", &FirstOrderDubinsMppiCostParams::steer_cmd_coeff)
    .def_readwrite("steer_rate_coeff", &FirstOrderDubinsMppiCostParams::steer_rate_coeff)
    .def_readwrite("steer_rate_l2_coeff", &FirstOrderDubinsMppiCostParams::steer_rate_l2_coeff)
    .def_readwrite("steer_accel_coeff", &FirstOrderDubinsMppiCostParams::steer_accel_coeff)
    .def_readwrite("cmd_slew_coeff", &FirstOrderDubinsMppiCostParams::cmd_slew_coeff)
    .def_readwrite(
      "nominal_curvature_min_chord_length_m",
      &FirstOrderDubinsMppiCostParams::nominal_curvature_min_chord_length_m)
    .def_readwrite(
      "lateral_acceleration_coeff", &FirstOrderDubinsMppiCostParams::lateral_acceleration_coeff)
    .def_readwrite("lateral_jerk_coeff", &FirstOrderDubinsMppiCostParams::lateral_jerk_coeff)
    .def_readwrite(
      "longitudinal_jerk_coeff", &FirstOrderDubinsMppiCostParams::longitudinal_jerk_coeff)
    .def_readwrite(
      "obstacle_collision_margin", &FirstOrderDubinsMppiCostParams::obstacle_collision_margin)
    .def_readwrite(
      "road_border_collision_margin", &FirstOrderDubinsMppiCostParams::road_border_collision_margin)
    .def_readwrite(
      "drivable_area_crossing_coeff", &FirstOrderDubinsMppiCostParams::drivable_area_crossing_coeff)
    .def_readwrite("goal_pos_coeff", &FirstOrderDubinsMppiCostParams::goal_pos_coeff)
    .def_readwrite("goal_speed_coeff", &FirstOrderDubinsMppiCostParams::goal_speed_coeff)
    .def_readwrite("goal_yaw_coeff", &FirstOrderDubinsMppiCostParams::goal_yaw_coeff)
    .def_readwrite("goal_terminal_scale", &FirstOrderDubinsMppiCostParams::goal_terminal_scale);

  py::class_<FirstOrderDubinsMppiRuntimeOptions>(module, "RuntimeOptions")
    .def(py::init<>())
    .def_readwrite(
      "enable_debug_trajectory_log",
      &FirstOrderDubinsMppiRuntimeOptions::enable_debug_trajectory_log)
    .def_readwrite(
      "debug_trajectory_log_directory",
      &FirstOrderDubinsMppiRuntimeOptions::debug_trajectory_log_directory)
    .def_readwrite("ignore_obstacles", &FirstOrderDubinsMppiRuntimeOptions::ignore_obstacles)
    .def_readwrite(
      "ignore_drivable_area", &FirstOrderDubinsMppiRuntimeOptions::ignore_drivable_area)
    .def_readwrite(
      "force_cold_start_each_step", &FirstOrderDubinsMppiRuntimeOptions::force_cold_start_each_step)
    .def_readwrite("skip_if_invalid", &FirstOrderDubinsMppiRuntimeOptions::skip_if_invalid)
    .def_readwrite(
      "use_last_control_as_nominal",
      &FirstOrderDubinsMppiRuntimeOptions::use_last_control_as_nominal);

  py::class_<FirstOrderDubinsMppiVehicleParams>(module, "VehicleParams")
    .def(py::init<>())
    .def_readwrite("ego_length", &FirstOrderDubinsMppiVehicleParams::ego_length)
    .def_readwrite("ego_width", &FirstOrderDubinsMppiVehicleParams::ego_width)
    .def_readwrite(
      "ego_axle_to_box_center", &FirstOrderDubinsMppiVehicleParams::ego_axle_to_box_center)
    .def_readwrite("wheel_base", &FirstOrderDubinsMppiVehicleParams::wheel_base)
    .def_readwrite("max_steer_angle", &FirstOrderDubinsMppiVehicleParams::max_steer_angle)
    .def_readwrite("acc_time_constant", &FirstOrderDubinsMppiVehicleParams::acc_time_constant)
    .def_readwrite("steer_time_constant", &FirstOrderDubinsMppiVehicleParams::steer_time_constant)
    .def_readwrite("steer_rate_lim", &FirstOrderDubinsMppiVehicleParams::steer_rate_lim)
    .def_readwrite("vel_rate_lim", &FirstOrderDubinsMppiVehicleParams::vel_rate_lim)
    .def_readwrite("acc_time_delay", &FirstOrderDubinsMppiVehicleParams::acc_time_delay)
    .def_readwrite("steer_time_delay", &FirstOrderDubinsMppiVehicleParams::steer_time_delay);

  py::class_<MppiConfiguration>(module, "Configuration")
    .def(py::init<>())
    .def_readwrite("name", &MppiConfiguration::name)
    .def_readwrite("cost_params", &MppiConfiguration::cost_params)
    .def_readwrite("runtime_options", &MppiConfiguration::runtime_options)
    .def_readwrite("vehicle_params", &MppiConfiguration::vehicle_params)
    .def_readwrite("boundary_search_margin_m", &MppiConfiguration::boundary_search_margin_m);

  py::class_<MppiEvaluationSession>(module, "EvaluationSession")
    .def(
      py::init([](
                 const py::bytes & map, const py::bytes & route,
                 const MppiConfiguration & configuration, const std::string & mode) {
        MppiEnvironment environment;
        environment.lanelet_map = deserialize_message<autoware_map_msgs::msg::LaneletMapBin>(map);
        environment.route = deserialize_message<autoware_planning_msgs::msg::LaneletRoute>(route);
        return std::make_unique<MppiEvaluationSession>(
          std::move(environment), configuration, parse_mode(mode));
      }),
      py::arg("lanelet_map_cdr"), py::arg("route_cdr"), py::arg("configuration"),
      py::arg("mode") = "isolated")
    .def("reset", &MppiEvaluationSession::reset)
    .def(
      "evaluate",
      [](
        MppiEvaluationSession & session, const std::string & frame_id,
        const std::uint64_t timestamp_ns, const py::bytes & trajectory_cdr,
        const py::bytes & odometry_cdr, const py::bytes & tracked_objects_cdr,
        const py::object & acceleration_cdr, const py::object & steering_cdr,
        const double odometry_age_ms, const py::object & acceleration_age_ms,
        const py::object & steering_age_ms, const py::object & tracked_objects_age_ms) {
        MppiInputFrame frame;
        frame.frame_id = frame_id;
        frame.timestamp_ns = timestamp_ns;
        frame.reference_trajectory =
          deserialize_message<autoware_planning_msgs::msg::Trajectory>(trajectory_cdr);
        frame.odometry = deserialize_message<nav_msgs::msg::Odometry>(odometry_cdr);
        frame.acceleration =
          deserialize_optional_message<geometry_msgs::msg::AccelWithCovarianceStamped>(
            acceleration_cdr);
        frame.steering_status =
          deserialize_optional_message<autoware_vehicle_msgs::msg::SteeringReport>(steering_cdr);
        frame.tracked_objects =
          deserialize_message<autoware_perception_msgs::msg::TrackedObjects>(tracked_objects_cdr);
        frame.odometry_age_ms = odometry_age_ms;
        if (!acceleration_age_ms.is_none()) {
          frame.acceleration_age_ms = acceleration_age_ms.cast<double>();
        }
        if (!steering_age_ms.is_none()) {
          frame.steering_age_ms = steering_age_ms.cast<double>();
        }
        if (!tracked_objects_age_ms.is_none()) {
          frame.tracked_objects_age_ms = tracked_objects_age_ms.cast<double>();
        }

        EvaluatedFrameResult result;
        {
          py::gil_scoped_release release;
          result = session.evaluate(frame);
        }
        return result_to_dict(result);
      },
      py::arg("frame_id"), py::arg("timestamp_ns"), py::arg("trajectory_cdr"),
      py::arg("odometry_cdr"), py::arg("tracked_objects_cdr"),
      py::arg("acceleration_cdr") = py::none(), py::arg("steering_cdr") = py::none(),
      py::arg("odometry_age_ms") = 0.0, py::arg("acceleration_age_ms") = py::none(),
      py::arg("steering_age_ms") = py::none(), py::arg("tracked_objects_age_ms") = py::none());
}
