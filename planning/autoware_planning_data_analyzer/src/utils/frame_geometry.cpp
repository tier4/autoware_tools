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

#include "utils/frame_geometry.hpp"

#include <autoware_vehicle_msgs/msg/control_mode_report.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_traffic_rules/TrafficRules.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace autoware::planning_data_analyzer::utils
{

namespace
{

// Arc length distance between the three points used for the route centerline curvature [m].
constexpr double kRouteCurvatureBaselineM = 3.0;
constexpr double kDuplicatePointThresholdM = 1.0e-3;

struct RawPoint
{
  double x;
  double y;
  int64_t lanelet_id;
};

struct SegmentSearchResult
{
  double distance{std::numeric_limits<double>::infinity()};
  RouteProjection projection;
};

SegmentSearchResult search_segments(
  const RouteCenterline & route, const double x, const double y, const std::size_t first_segment,
  const std::size_t last_segment)
{
  SegmentSearchResult best;
  for (std::size_t j = first_segment; j <= last_segment; ++j) {
    const double ax = route.x[j];
    const double ay = route.y[j];
    const double dx = route.x[j + 1] - ax;
    const double dy = route.y[j + 1] - ay;
    const double length_sq = dx * dx + dy * dy;
    const double ratio =
      length_sq > 0.0 ? std::clamp(((x - ax) * dx + (y - ay) * dy) / length_sq, 0.0, 1.0) : 0.0;
    const double foot_x = ax + ratio * dx;
    const double foot_y = ay + ratio * dy;
    const double distance = std::hypot(x - foot_x, y - foot_y);
    if (distance < best.distance) {
      const double length = std::sqrt(length_sq);
      const double cross = length > 0.0 ? (dx * (y - ay) - dy * (x - ax)) / length : 0.0;
      best.distance = distance;
      best.projection.s = route.s[j] + ratio * (route.s[j + 1] - route.s[j]);
      best.projection.lateral_offset = std::copysign(distance, cross);
      best.projection.index = ratio < 0.5 ? j : j + 1;
    }
  }
  return best;
}

}  // namespace

RouteCenterline build_route_centerline(
  const std::vector<RouteLaneletSource> & lanelets, const double resample_interval_m)
{
  RouteCenterline route;
  route.resample_interval_m = resample_interval_m;
  if (resample_interval_m <= 0.0) {
    return route;
  }

  // Concatenate lanelet centerlines. The segment ending at raw[k] belongs to raw[k].lanelet_id,
  // so a gap between lanelets is attributed to the following lanelet.
  std::vector<RawPoint> raw;
  std::vector<double> raw_s;
  for (const auto & lanelet : lanelets) {
    RouteLaneletSection section;
    section.id = lanelet.id;
    section.s_start = raw_s.empty() ? 0.0 : raw_s.back();
    section.turn_direction = lanelet.turn_direction.empty() ? "none" : lanelet.turn_direction;
    section.is_intersection = section.turn_direction != "none";
    section.speed_limit_mps = lanelet.speed_limit_mps;

    for (const auto & [x, y] : lanelet.points) {
      if (!raw.empty()) {
        const double step = std::hypot(x - raw.back().x, y - raw.back().y);
        if (step < kDuplicatePointThresholdM) {
          continue;
        }
        raw_s.push_back(raw_s.back() + step);
      } else {
        raw_s.push_back(0.0);
      }
      raw.push_back({x, y, lanelet.id});
    }
    section.s_end = raw_s.empty() ? 0.0 : raw_s.back();
    route.lanelets.push_back(section);
  }
  if (raw.size() < 2) {
    route.lanelets.clear();
    return route;
  }

  const double total_length = raw_s.back();
  std::vector<double> sample_s;
  for (double s = 0.0; s < total_length; s += resample_interval_m) {
    sample_s.push_back(s);
  }
  if (total_length - sample_s.back() > kDuplicatePointThresholdM) {
    sample_s.push_back(total_length);
  }

  std::size_t segment = 0;
  for (const double s : sample_s) {
    while (segment + 2 < raw.size() && raw_s[segment + 1] < s) {
      ++segment;
    }
    const auto & begin = raw[segment];
    const auto & end = raw[segment + 1];
    const double segment_length = raw_s[segment + 1] - raw_s[segment];
    const double ratio =
      segment_length > 0.0 ? std::clamp((s - raw_s[segment]) / segment_length, 0.0, 1.0) : 0.0;
    route.s.push_back(s);
    route.x.push_back(begin.x + ratio * (end.x - begin.x));
    route.y.push_back(begin.y + ratio * (end.y - begin.y));
    route.yaw.push_back(std::atan2(end.y - begin.y, end.x - begin.x));
    route.lanelet_id.push_back(end.lanelet_id);
  }
  route.curvature = calc_signed_curvature(route.x, route.y, kRouteCurvatureBaselineM);
  return route;
}

namespace
{

RouteLaneletSource to_lanelet_source(
  const lanelet::ConstLanelet & lanelet, const lanelet::traffic_rules::TrafficRulesPtr & rules)
{
  RouteLaneletSource source;
  source.id = lanelet.id();
  for (const auto & point : lanelet.centerline()) {
    source.points.emplace_back(point.x(), point.y());
  }
  source.turn_direction = lanelet.attributeOr("turn_direction", "none");
  if (rules) {
    try {
      source.speed_limit_mps = rules->speedLimit(lanelet).speedLimit.value();
    } catch (const std::exception &) {
      source.speed_limit_mps = std::nullopt;
    }
  }
  return source;
}

}  // namespace

std::vector<RouteLaneletSource> route_lanelet_sources(
  const autoware::route_handler::RouteHandler & map_handler,
  const autoware_planning_msgs::msg::LaneletRoute & route)
{
  std::vector<RouteLaneletSource> sources;
  const auto map = map_handler.getLaneletMapPtr();
  if (!map) {
    return sources;
  }
  const auto rules = map_handler.getTrafficRulesPtr();
  for (const auto & segment : route.segments) {
    const auto id = segment.preferred_primitive.id;
    if (!map->laneletLayer.exists(id)) {
      continue;
    }
    sources.push_back(to_lanelet_source(map->laneletLayer.get(id), rules));
  }
  return sources;
}

std::vector<RouteLaneletSource> route_lanelet_sources(
  const autoware::route_handler::RouteHandler & route_handler)
{
  std::vector<RouteLaneletSource> sources;
  if (!route_handler.isHandlerReady()) {
    return sources;
  }
  const auto rules = route_handler.getTrafficRulesPtr();
  for (const auto & lanelet : route_handler.getPreferredLanelets()) {
    sources.push_back(to_lanelet_source(lanelet, rules));
  }
  return sources;
}

namespace
{

// Search segments [first_segment, last_segment], around the hint first.
std::optional<RouteProjection> project_in_range(
  const RouteCenterline & route, const double x, const double y, const double max_lateral_offset_m,
  const std::size_t first_segment, const std::size_t last_segment,
  const std::optional<std::size_t> hint, const double search_backward_m,
  const double search_forward_m)
{
  if (hint.has_value() && route.resample_interval_m > 0.0) {
    const auto backward = static_cast<std::size_t>(
      std::ceil(std::max(search_backward_m, 0.0) / route.resample_interval_m));
    const auto forward = static_cast<std::size_t>(
      std::ceil(std::max(search_forward_m, 0.0) / route.resample_interval_m));
    const std::size_t center = std::clamp(*hint, first_segment, last_segment);
    const std::size_t first = std::max(first_segment, center > backward ? center - backward : 0);
    const std::size_t last = std::min(center + forward, last_segment);
    const auto local = search_segments(route, x, y, first, last);
    if (local.distance <= max_lateral_offset_m) {
      return local.projection;
    }
  }

  const auto whole = search_segments(route, x, y, first_segment, last_segment);
  if (whole.distance <= max_lateral_offset_m) {
    return whole.projection;
  }
  return std::nullopt;
}

// Index of the route leg active at stamp_ns (the first leg before its stamp).
std::size_t find_active_leg(
  const std::vector<RouteLegSection> & legs, const rcutils_time_point_value_t stamp_ns)
{
  const auto it = std::upper_bound(
    legs.begin(), legs.end(), stamp_ns,
    [](const rcutils_time_point_value_t t, const RouteLegSection & leg) {
      return t < leg.active_from_ns;
    });
  return it == legs.begin() ? 0 : static_cast<std::size_t>(std::distance(legs.begin(), it) - 1);
}

}  // namespace

std::optional<RouteProjection> project_to_route(
  const RouteCenterline & route, const double x, const double y, const double max_lateral_offset_m,
  const std::optional<std::size_t> hint, const double search_backward_m,
  const double search_forward_m)
{
  if (route.s.size() < 2) {
    return std::nullopt;
  }
  return project_in_range(
    route, x, y, max_lateral_offset_m, 0, route.s.size() - 2, hint, search_backward_m,
    search_forward_m);
}

RouteCenterline build_drive_route(
  const std::vector<RouteLegInput> & legs, const std::vector<TimedPoint> & samples,
  const double max_lateral_offset_m, const double clip_margin_m)
{
  RouteCenterline drive;
  if (legs.empty()) {
    return drive;
  }
  drive.resample_interval_m = legs.front().centerline.resample_interval_m;
  for (std::size_t k = 0; k < legs.size(); ++k) {
    RouteLegSection section;
    section.index = static_cast<int>(k);
    section.uuid = legs[k].uuid;
    section.active_from_ns = legs[k].active_from_ns;
    drive.routes.push_back(section);
  }

  // Samples of each leg in time order
  std::vector<TimedPoint> sorted_samples = samples;
  std::sort(sorted_samples.begin(), sorted_samples.end(), [](const auto & a, const auto & b) {
    return a.stamp_ns < b.stamp_ns;
  });
  std::vector<std::vector<TimedPoint>> leg_samples(legs.size());
  for (const auto & sample : sorted_samples) {
    leg_samples[find_active_leg(drive.routes, sample.stamp_ns)].push_back(sample);
  }

  for (std::size_t k = 0; k < legs.size(); ++k) {
    const auto & leg = legs[k].centerline;
    if (leg.s.size() < 2) {
      continue;
    }

    // Arc length range of the leg that the ego drove while the leg was active
    std::optional<double> s_min;
    std::optional<double> s_max;
    std::optional<std::size_t> hint;
    for (const auto & sample : leg_samples[k]) {
      const auto projection = project_to_route(leg, sample.x, sample.y, max_lateral_offset_m, hint);
      if (!projection) {
        continue;
      }
      hint = projection->index;
      s_min = std::min(s_min.value_or(projection->s), projection->s);
      s_max = std::max(s_max.value_or(projection->s), projection->s);
    }
    if (!s_min) {
      continue;
    }
    const double clip_start = std::max(0.0, *s_min - clip_margin_m);
    const double clip_end = std::min(leg.length(), *s_max + clip_margin_m);
    std::size_t first = 0;
    while (first + 1 < leg.s.size() && leg.s[first + 1] <= clip_start) {
      ++first;
    }
    std::size_t last = leg.s.size() - 1;
    while (last > first && leg.s[last - 1] >= clip_end) {
      --last;
    }

    const double offset =
      drive.s.empty()
        ? -leg.s[first]
        : drive.s.back() +
            std::hypot(leg.x[first] - drive.x.back(), leg.y[first] - drive.y.back()) - leg.s[first];
    auto & section = drive.routes[k];
    section.used = true;
    section.first_point = drive.s.size();
    for (std::size_t i = first; i <= last; ++i) {
      drive.s.push_back(leg.s[i] + offset);
      drive.x.push_back(leg.x[i]);
      drive.y.push_back(leg.y[i]);
      drive.yaw.push_back(leg.yaw[i]);
      drive.curvature.push_back(leg.curvature[i]);
      drive.lanelet_id.push_back(leg.lanelet_id[i]);
      drive.route_index.push_back(static_cast<int>(k));
    }
    section.last_point = drive.s.size() - 1;
    section.s_start = drive.s[section.first_point];
    section.s_end = drive.s[section.last_point];

    for (const auto & lanelet : leg.lanelets) {
      const double start = std::max(lanelet.s_start, leg.s[first]);
      const double end = std::min(lanelet.s_end, leg.s[last]);
      if (end <= start) {
        continue;
      }
      auto clipped = lanelet;
      clipped.route_index = static_cast<int>(k);
      clipped.s_start = start + offset;
      clipped.s_end = end + offset;
      drive.lanelets.push_back(clipped);
    }
  }
  return drive;
}

std::optional<RouteProjection> project_to_drive_route(
  const RouteCenterline & route, const rcutils_time_point_value_t stamp_ns, const double x,
  const double y, const double max_lateral_offset_m, const std::optional<std::size_t> hint)
{
  if (route.routes.empty()) {
    return project_to_route(route, x, y, max_lateral_offset_m, hint);
  }
  const auto & leg = route.routes[find_active_leg(route.routes, stamp_ns)];
  if (!leg.used || leg.last_point <= leg.first_point) {
    return std::nullopt;
  }
  return project_in_range(
    route, x, y, max_lateral_offset_m, leg.first_point, leg.last_point - 1, hint, 50.0, 200.0);
}

std::vector<std::optional<double>> calc_signed_curvature(
  const std::vector<double> & x, const std::vector<double> & y, const double baseline_m)
{
  const std::size_t n = std::min(x.size(), y.size());
  std::vector<std::optional<double>> curvature(n);
  if (n < 3) {
    return curvature;
  }

  std::vector<double> arc(n, 0.0);
  for (std::size_t i = 1; i < n; ++i) {
    arc[i] = arc[i - 1] + std::hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  }

  std::size_t prev = 0;
  std::size_t next = 0;
  for (std::size_t i = 0; i < n; ++i) {
    // prev: the last index whose distance to i is still >= baseline
    while (prev + 1 < i && arc[i] - arc[prev + 1] >= baseline_m) {
      ++prev;
    }
    next = std::max(next, i + 1);
    while (next < n && arc[next] - arc[i] < baseline_m) {
      ++next;
    }
    if (i == 0 || arc[i] - arc[prev] < baseline_m || next >= n) {
      continue;
    }

    const double ax = x[prev];
    const double ay = y[prev];
    const double bx = x[i];
    const double by = y[i];
    const double cx = x[next];
    const double cy = y[next];
    const double ab = std::hypot(bx - ax, by - ay);
    const double bc = std::hypot(cx - bx, cy - by);
    const double ca = std::hypot(ax - cx, ay - cy);
    const double denominator = ab * bc * ca;
    if (denominator <= 0.0) {
      continue;
    }
    const double cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    curvature[i] = 2.0 * cross / denominator;
  }

  // Points near both ends take the value of the nearest point that has one.
  const auto first_valid = std::find_if(
    curvature.begin(), curvature.end(), [](const auto & value) { return value.has_value(); });
  if (first_valid == curvature.end()) {
    return curvature;
  }
  std::fill(curvature.begin(), first_valid, *first_valid);
  const auto last_valid = std::find_if(
    curvature.rbegin(), curvature.rend(), [](const auto & value) { return value.has_value(); });
  std::fill(curvature.rbegin(), last_valid, *last_valid);
  return curvature;
}

std::optional<uint8_t> find_control_mode(
  const std::vector<ControlModeEvent> & sorted_events, const rcutils_time_point_value_t stamp_ns)
{
  const auto it = std::upper_bound(
    sorted_events.begin(), sorted_events.end(), stamp_ns,
    [](const rcutils_time_point_value_t t, const ControlModeEvent & event) {
      return t < event.first;
    });
  if (it == sorted_events.begin()) {
    return std::nullopt;
  }
  return std::prev(it)->second;
}

std::string control_mode_name(const std::optional<uint8_t> mode)
{
  using autoware_vehicle_msgs::msg::ControlModeReport;
  if (!mode.has_value()) {
    return "UNKNOWN";
  }
  switch (*mode) {
    case ControlModeReport::NO_COMMAND:
      return "NO_COMMAND";
    case ControlModeReport::AUTONOMOUS:
      return "AUTONOMOUS";
    case ControlModeReport::AUTONOMOUS_STEER_ONLY:
      return "AUTONOMOUS_STEER_ONLY";
    case ControlModeReport::AUTONOMOUS_VELOCITY_ONLY:
      return "AUTONOMOUS_VELOCITY_ONLY";
    case ControlModeReport::MANUAL:
      return "MANUAL";
    case ControlModeReport::DISENGAGED:
      return "DISENGAGED";
    case ControlModeReport::NOT_READY:
      return "NOT_READY";
    default:
      return "UNKNOWN";
  }
}

nlohmann::json rounded_json(const std::optional<double> value, const int decimals)
{
  if (!value.has_value() || !std::isfinite(*value)) {
    return nullptr;
  }
  const double factor = std::pow(10.0, decimals);
  const double rounded = std::round(*value * factor) / factor;
  return rounded == 0.0 ? 0.0 : rounded;  // avoid "-0.0"
}

nlohmann::json rounded_json_array(const std::vector<double> & values, const int decimals)
{
  nlohmann::json array = nlohmann::json::array();
  for (const double value : values) {
    array.push_back(rounded_json(value, decimals));
  }
  return array;
}

nlohmann::json rounded_json_array(
  const std::vector<std::optional<double>> & values, const int decimals)
{
  nlohmann::json array = nlohmann::json::array();
  for (const auto & value : values) {
    array.push_back(rounded_json(value, decimals));
  }
  return array;
}

nlohmann::json route_centerline_to_json(
  const RouteCenterline & route, const FrameGeometryParameters & params)
{
  nlohmann::json j;
  j["frame_id"] = "map";
  j["resample_interval_m"] = route.resample_interval_m;
  j["length_m"] = rounded_json(route.length(), params.position_decimals);

  auto & centerline = j["centerline"];
  centerline["s"] = rounded_json_array(route.s, params.position_decimals);
  centerline["x"] = rounded_json_array(route.x, params.position_decimals);
  centerline["y"] = rounded_json_array(route.y, params.position_decimals);
  centerline["yaw"] = rounded_json_array(route.yaw, params.angle_decimals);
  centerline["curvature"] = rounded_json_array(route.curvature, params.curvature_decimals);
  centerline["lanelet_id"] = route.lanelet_id;
  centerline["route_index"] = route.route_index;

  j["routes"] = nlohmann::json::array();
  for (const auto & leg : route.routes) {
    j["routes"].push_back(
      {{"index", leg.index},
       {"uuid", leg.uuid},
       {"active_from_sec", static_cast<double>(leg.active_from_ns) * 1.0e-9},
       {"used", leg.used},
       {"s_start",
        leg.used ? rounded_json(leg.s_start, params.position_decimals) : nlohmann::json(nullptr)},
       {"s_end",
        leg.used ? rounded_json(leg.s_end, params.position_decimals) : nlohmann::json(nullptr)}});
  }

  j["lanelets"] = nlohmann::json::array();
  for (const auto & section : route.lanelets) {
    j["lanelets"].push_back(
      {{"id", section.id},
       {"route_index", section.route_index},
       {"s_start", rounded_json(section.s_start, params.position_decimals)},
       {"s_end", rounded_json(section.s_end, params.position_decimals)},
       {"turn_direction", section.turn_direction},
       {"is_intersection", section.is_intersection},
       {"speed_limit_mps", rounded_json(section.speed_limit_mps, 3)}});
  }
  return j;
}

}  // namespace autoware::planning_data_analyzer::utils
