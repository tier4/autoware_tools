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

#ifndef UTILS__FRAME_GEOMETRY_HPP_
#define UTILS__FRAME_GEOMETRY_HPP_

#include "utils/override_windows.hpp"

#include <autoware/route_handler/route_handler.hpp>
#include <nlohmann/json.hpp>

#include <autoware_planning_msgs/msg/lanelet_route.hpp>

#include <rcutils/time.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::planning_data_analyzer::utils
{

/**
 * @brief Parameters of the frame geometry export (frame_geometry.jsonl and route.json).
 */
struct FrameGeometryParameters
{
  bool enabled{false};
  int position_decimals{3};
  int angle_decimals{4};
  int curvature_decimals{5};
  double route_resample_interval_m{1.0};
  double max_route_lateral_offset_m{5.0};
};

/**
 * @brief Centerline of one route lanelet used as input of the route centerline.
 */
struct RouteLaneletSource
{
  int64_t id{0};
  std::vector<std::pair<double, double>> points;  // (x, y) in map frame
  std::string turn_direction{"none"};
  std::optional<double> speed_limit_mps;
};

/**
 * @brief Arc length range of one lanelet on the route centerline.
 */
struct RouteLaneletSection
{
  int64_t id{0};
  int route_index{0};
  double s_start{0.0};
  double s_end{0.0};
  std::string turn_direction{"none"};
  bool is_intersection{false};
  std::optional<double> speed_limit_mps;
};

/**
 * @brief One route message (leg) of a drive route.
 *
 * A route is active from its stamp until the stamp of the next route. Only the part of the
 * route that the ego actually drove while it was active is kept in the drive route
 * ([first_point, last_point] of RouteCenterline). used is false if the ego was never on it.
 */
struct RouteLegSection
{
  int index{0};
  std::string uuid;
  rcutils_time_point_value_t active_from_ns{0};
  bool used{false};
  double s_start{0.0};
  double s_end{0.0};
  std::size_t first_point{0};
  std::size_t last_point{0};
};

/**
 * @brief Route centerline resampled at a constant arc length interval.
 *
 * curvature is signed (left turn is positive) [1/m]; it is std::nullopt where it cannot be
 * computed (e.g. the route is shorter than the curvature baseline).
 * route_index / routes are filled only for a drive route (build_drive_route()).
 */
struct RouteCenterline
{
  double resample_interval_m{1.0};
  std::vector<double> s;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> yaw;
  std::vector<std::optional<double>> curvature;
  std::vector<int64_t> lanelet_id;
  std::vector<int> route_index;
  std::vector<RouteLaneletSection> lanelets;
  std::vector<RouteLegSection> routes;

  bool empty() const { return s.empty(); }
  double length() const { return s.empty() ? 0.0 : s.back(); }
};

/**
 * @brief Result of projecting a point onto the route centerline.
 */
struct RouteProjection
{
  double s{0.0};               // arc length of the foot of the perpendicular [m]
  double lateral_offset{0.0};  // signed distance from the centerline, left is positive [m]
  std::size_t index{0};        // index of the nearest resampled centerline point
};

/**
 * @brief Concatenate lanelet centerlines in route order and resample them.
 *
 * Gaps between consecutive lanelets (e.g. lane changes between preferred lanelets) are
 * bridged by a straight segment that is attributed to the following lanelet.
 */
RouteCenterline build_route_centerline(
  const std::vector<RouteLaneletSource> & lanelets, double resample_interval_m);

/**
 * @brief Preferred lanelets of a route message, looked up in the map of @p map_handler.
 */
std::vector<RouteLaneletSource> route_lanelet_sources(
  const autoware::route_handler::RouteHandler & map_handler,
  const autoware_planning_msgs::msg::LaneletRoute & route);

/**
 * @brief Preferred lanelets of the route currently set to @p route_handler.
 */
std::vector<RouteLaneletSource> route_lanelet_sources(
  const autoware::route_handler::RouteHandler & route_handler);

/**
 * @brief Input of build_drive_route(): one route message and the time it became active.
 */
struct RouteLegInput
{
  RouteCenterline centerline;
  rcutils_time_point_value_t active_from_ns{0};
  std::string uuid;
};

/**
 * @brief Ego position at a time, used to find which part of each route was driven.
 */
struct TimedPoint
{
  rcutils_time_point_value_t stamp_ns{0};
  double x{0.0};
  double y{0.0};
};

/**
 * @brief Join the routes of a drive into one centerline in time order.
 *
 * A drive often consists of several routes (e.g. a bus going from stop to stop gets a new
 * route at each stop). Each sample is assigned to the route active at its stamp (the first
 * route is also used before its stamp), and each route is clipped to the arc length range of
 * its samples (+/- @p clip_margin_m). The clipped routes are concatenated with continuous
 * arc length; a gap between them is counted by its straight-line distance.
 *
 * @param legs Routes sorted by active_from_ns.
 */
RouteCenterline build_drive_route(
  const std::vector<RouteLegInput> & legs, const std::vector<TimedPoint> & samples,
  double max_lateral_offset_m, double clip_margin_m = 0.0);

/**
 * @brief Project a point at @p stamp_ns onto the part of the drive route that belongs to the
 * route active at that time. See project_to_route() for @p hint.
 */
std::optional<RouteProjection> project_to_drive_route(
  const RouteCenterline & route, rcutils_time_point_value_t stamp_ns, double x, double y,
  double max_lateral_offset_m, std::optional<std::size_t> hint = std::nullopt);

/**
 * @brief Project a point onto the route centerline.
 *
 * If @p hint is given, the segments within [-search_backward_m, +search_forward_m] of the hint
 * are searched first, and the whole centerline is searched only when that fails. This keeps
 * the projection on the same pass when the route goes through the same place twice.
 *
 * @return std::nullopt if the centerline is empty or the point is farther than
 *         @p max_lateral_offset_m from it.
 */
std::optional<RouteProjection> project_to_route(
  const RouteCenterline & route, double x, double y, double max_lateral_offset_m,
  std::optional<std::size_t> hint = std::nullopt, double search_backward_m = 50.0,
  double search_forward_m = 200.0);

/**
 * @brief Signed curvature of a polyline by three points about @p baseline_m apart along it.
 *
 * For each point, the previous / next points are the first ones whose arc length distance is
 * at least @p baseline_m. Points without such neighbors take the value of the nearest point
 * that has one. All values are std::nullopt if no point has both neighbors (e.g. a stopped
 * trajectory).
 */
std::vector<std::optional<double>> calc_signed_curvature(
  const std::vector<double> & x, const std::vector<double> & y, double baseline_m);

/**
 * @brief ControlModeReport mode at @p stamp_ns (the last event at or before it).
 */
std::optional<uint8_t> find_control_mode(
  const std::vector<ControlModeEvent> & sorted_events, rcutils_time_point_value_t stamp_ns);

/**
 * @brief Name of a ControlModeReport mode, e.g. "AUTONOMOUS". "UNKNOWN" for std::nullopt.
 */
std::string control_mode_name(std::optional<uint8_t> mode);

/**
 * @brief JSON number rounded to @p decimals digits, or null if not finite / std::nullopt.
 */
nlohmann::json rounded_json(std::optional<double> value, int decimals);

/**
 * @brief JSON array of rounded numbers (null for non-finite / std::nullopt elements).
 */
nlohmann::json rounded_json_array(const std::vector<double> & values, int decimals);
nlohmann::json rounded_json_array(const std::vector<std::optional<double>> & values, int decimals);

/**
 * @brief Serialize the route centerline for time_step_based_trajectory_route.json.
 */
nlohmann::json route_centerline_to_json(
  const RouteCenterline & route, const FrameGeometryParameters & params);

}  // namespace autoware::planning_data_analyzer::utils

#endif  // UTILS__FRAME_GEOMETRY_HPP_
