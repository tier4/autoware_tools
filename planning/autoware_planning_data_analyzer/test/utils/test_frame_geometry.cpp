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

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace autoware::planning_data_analyzer::utils
{
namespace
{

RouteLaneletSource make_lanelet(
  const int64_t id, const std::vector<std::pair<double, double>> & points,
  const std::string & turn_direction = "none")
{
  RouteLaneletSource lanelet;
  lanelet.id = id;
  lanelet.points = points;
  lanelet.turn_direction = turn_direction;
  return lanelet;
}

// Two straight lanelets along +x: [0, 10] and [10, 20].
RouteCenterline make_straight_route()
{
  return build_route_centerline(
    {make_lanelet(1, {{0.0, 0.0}, {10.0, 0.0}}),
     make_lanelet(2, {{10.0, 0.0}, {20.0, 0.0}}, "left")},
    1.0);
}

}  // namespace

TEST(FrameGeometryTest, BuildRouteCenterlineResamplesAndKeepsLaneletRanges)
{
  const auto route = make_straight_route();

  ASSERT_EQ(route.s.size(), 21U);
  EXPECT_DOUBLE_EQ(route.length(), 20.0);
  EXPECT_DOUBLE_EQ(route.s[5], 5.0);
  EXPECT_DOUBLE_EQ(route.x[5], 5.0);
  EXPECT_DOUBLE_EQ(route.y[5], 0.0);
  EXPECT_DOUBLE_EQ(route.yaw[5], 0.0);
  EXPECT_EQ(route.lanelet_id[5], 1);
  EXPECT_EQ(route.lanelet_id[15], 2);

  ASSERT_EQ(route.lanelets.size(), 2U);
  EXPECT_DOUBLE_EQ(route.lanelets[0].s_start, 0.0);
  EXPECT_DOUBLE_EQ(route.lanelets[0].s_end, 10.0);
  EXPECT_FALSE(route.lanelets[0].is_intersection);
  EXPECT_DOUBLE_EQ(route.lanelets[1].s_start, 10.0);
  EXPECT_DOUBLE_EQ(route.lanelets[1].s_end, 20.0);
  EXPECT_EQ(route.lanelets[1].turn_direction, "left");
  EXPECT_TRUE(route.lanelets[1].is_intersection);

  for (const auto & curvature : route.curvature) {
    ASSERT_TRUE(curvature.has_value());
    EXPECT_NEAR(*curvature, 0.0, 1.0e-9);
  }
}

TEST(FrameGeometryTest, BuildRouteCenterlineAddsLastPointForFractionalLength)
{
  const auto route = build_route_centerline({make_lanelet(1, {{0.0, 0.0}, {2.5, 0.0}})}, 1.0);
  ASSERT_EQ(route.s.size(), 4U);
  EXPECT_DOUBLE_EQ(route.s.back(), 2.5);
  EXPECT_DOUBLE_EQ(route.x.back(), 2.5);
}

TEST(FrameGeometryTest, BuildRouteCenterlineReturnsEmptyForDegenerateInput)
{
  EXPECT_TRUE(build_route_centerline(std::vector<RouteLaneletSource>{}, 1.0).empty());
  EXPECT_TRUE(build_route_centerline({make_lanelet(1, {{0.0, 0.0}})}, 1.0).empty());
  EXPECT_TRUE(build_route_centerline({make_lanelet(1, {{0.0, 0.0}, {1.0, 0.0}})}, 0.0).empty());
}

TEST(FrameGeometryTest, ProjectToRouteReturnsArcLengthAndSignedOffset)
{
  const auto route = make_straight_route();

  const auto left = project_to_route(route, 5.3, 2.0, 5.0);
  ASSERT_TRUE(left.has_value());
  EXPECT_NEAR(left->s, 5.3, 1.0e-9);
  EXPECT_NEAR(left->lateral_offset, 2.0, 1.0e-9);
  EXPECT_EQ(left->index, 5U);

  const auto right = project_to_route(route, 12.7, -1.5, 5.0);
  ASSERT_TRUE(right.has_value());
  EXPECT_NEAR(right->s, 12.7, 1.0e-9);
  EXPECT_NEAR(right->lateral_offset, -1.5, 1.0e-9);
  EXPECT_EQ(right->index, 13U);

  EXPECT_FALSE(project_to_route(route, 5.0, 6.0, 5.0).has_value());
  EXPECT_FALSE(project_to_route(route, 30.0, 0.0, 5.0).has_value());
  EXPECT_FALSE(project_to_route(RouteCenterline{}, 0.0, 0.0, 5.0).has_value());
}

TEST(FrameGeometryTest, ProjectToRouteWithHintStaysOnTheSamePass)
{
  // The route goes along +x at y=0 and comes back along -x at y=3 (a U-turn).
  const auto route = build_route_centerline(
    {make_lanelet(1, {{0.0, 0.0}, {100.0, 0.0}}), make_lanelet(2, {{100.0, 0.0}, {100.0, 3.0}}),
     make_lanelet(3, {{100.0, 3.0}, {0.0, 3.0}})},
    1.0);

  // A point at y=1.6 is slightly closer to the return pass, but the hint says the ego
  // is still on the outward pass.
  const auto with_hint = project_to_route(route, 50.0, 1.6, 5.0, 50U, 20.0, 20.0);
  ASSERT_TRUE(with_hint.has_value());
  EXPECT_NEAR(with_hint->s, 50.0, 1.0e-9);

  const auto without_hint = project_to_route(route, 50.0, 1.6, 5.0);
  ASSERT_TRUE(without_hint.has_value());
  EXPECT_NEAR(without_hint->s, 153.0, 1.0e-9);
}

TEST(FrameGeometryTest, CalcSignedCurvatureOfCircle)
{
  constexpr double radius = 20.0;
  std::vector<double> x;
  std::vector<double> y;
  for (int i = 0; i <= 60; ++i) {
    const double angle = 0.01 * i;
    x.push_back(radius * std::sin(angle));
    y.push_back(radius * (1.0 - std::cos(angle)));  // counterclockwise = left turn
  }

  const auto left = calc_signed_curvature(x, y, 1.0);
  ASSERT_EQ(left.size(), x.size());
  for (const auto & curvature : left) {
    ASSERT_TRUE(curvature.has_value());
    EXPECT_NEAR(*curvature, 1.0 / radius, 1.0e-6);
  }

  for (auto & value : y) {
    value = -value;  // mirror: right turn
  }
  const auto right = calc_signed_curvature(x, y, 1.0);
  for (const auto & curvature : right) {
    ASSERT_TRUE(curvature.has_value());
    EXPECT_NEAR(*curvature, -1.0 / radius, 1.0e-6);
  }
}

TEST(FrameGeometryTest, CalcSignedCurvatureOfStoppedTrajectoryIsNull)
{
  const std::vector<double> x(10, 3.0);
  const std::vector<double> y(10, 4.0);
  for (const auto & curvature : calc_signed_curvature(x, y, 1.0)) {
    EXPECT_FALSE(curvature.has_value());
  }
  EXPECT_EQ(calc_signed_curvature({0.0, 1.0}, {0.0, 0.0}, 1.0).size(), 2U);
}

TEST(FrameGeometryTest, FindControlModeUsesLastEventAtOrBeforeStamp)
{
  using autoware_vehicle_msgs::msg::ControlModeReport;
  const std::vector<ControlModeEvent> events = {
    {100, ControlModeReport::MANUAL}, {200, ControlModeReport::AUTONOMOUS}};

  EXPECT_FALSE(find_control_mode(events, 99).has_value());
  EXPECT_EQ(find_control_mode(events, 100), ControlModeReport::MANUAL);
  EXPECT_EQ(find_control_mode(events, 199), ControlModeReport::MANUAL);
  EXPECT_EQ(find_control_mode(events, 200), ControlModeReport::AUTONOMOUS);
  EXPECT_EQ(find_control_mode(events, 1000), ControlModeReport::AUTONOMOUS);

  EXPECT_EQ(control_mode_name(ControlModeReport::AUTONOMOUS), "AUTONOMOUS");
  EXPECT_EQ(control_mode_name(ControlModeReport::MANUAL), "MANUAL");
  EXPECT_EQ(control_mode_name(std::nullopt), "UNKNOWN");
}

TEST(FrameGeometryTest, RoundedJson)
{
  EXPECT_DOUBLE_EQ(rounded_json(1.23456, 3).get<double>(), 1.235);
  EXPECT_EQ(rounded_json(-0.0001, 3).dump(), "0.0");
  EXPECT_TRUE(rounded_json(std::nullopt, 3).is_null());
  EXPECT_TRUE(rounded_json(std::numeric_limits<double>::infinity(), 3).is_null());
  EXPECT_TRUE(rounded_json(std::numeric_limits<double>::quiet_NaN(), 3).is_null());
  EXPECT_EQ(rounded_json_array(std::vector<double>{0.12345, 1.0}, 2).dump(), "[0.12,1.0]");
}

TEST(FrameGeometryTest, RouteCenterlineToJson)
{
  const auto json = route_centerline_to_json(make_straight_route(), FrameGeometryParameters{});
  EXPECT_EQ(json["frame_id"], "map");
  EXPECT_DOUBLE_EQ(json["length_m"].get<double>(), 20.0);
  EXPECT_EQ(json["centerline"]["s"].size(), 21U);
  EXPECT_EQ(json["centerline"]["lanelet_id"][15], 2);
  ASSERT_EQ(json["lanelets"].size(), 2U);
  EXPECT_EQ(json["lanelets"][1]["turn_direction"], "left");
  EXPECT_TRUE(json["routes"].empty());
  EXPECT_TRUE(json["lanelets"][1]["speed_limit_mps"].is_null());
}

TEST(FrameGeometryTest, BuildDriveRouteClipsAndJoinsRoutesInTimeOrder)
{
  // Route A: x in [0, 100] (active from t=0), route B: x in [100, 200] (active from t=100).
  // The ego drives x=20..90 on A and x=100..140 on B.
  std::vector<RouteLegInput> legs;
  legs.push_back(
    {build_route_centerline({make_lanelet(1, {{0.0, 0.0}, {100.0, 0.0}})}, 1.0), 0, "a"});
  legs.push_back(
    {build_route_centerline({make_lanelet(2, {{100.0, 0.0}, {200.0, 0.0}}, "right")}, 1.0), 100,
     "b"});
  std::vector<TimedPoint> samples;
  for (int t = 50; t < 100; ++t) {
    samples.push_back({t, 20.0 + (t - 50) * 70.0 / 49.0, 0.0});
  }
  for (int t = 100; t <= 140; ++t) {
    samples.push_back({t, 100.0 + (t - 100), 0.0});
  }

  const auto drive = build_drive_route(legs, samples, 5.0, 5.0);
  ASSERT_EQ(drive.routes.size(), 2U);
  EXPECT_TRUE(drive.routes[0].used);
  EXPECT_TRUE(drive.routes[1].used);
  EXPECT_DOUBLE_EQ(drive.x.front(), 15.0);
  EXPECT_DOUBLE_EQ(drive.s.front(), 0.0);
  EXPECT_DOUBLE_EQ(drive.routes[0].s_end, 80.0);    // x=95
  EXPECT_DOUBLE_EQ(drive.routes[1].s_start, 85.0);  // x=100, the 5 m gap is counted
  EXPECT_DOUBLE_EQ(drive.x.back(), 145.0);
  EXPECT_DOUBLE_EQ(drive.length(), 130.0);
  EXPECT_EQ(drive.route_index.front(), 0);
  EXPECT_EQ(drive.route_index.back(), 1);
  ASSERT_EQ(drive.lanelets.size(), 2U);
  EXPECT_EQ(drive.lanelets[1].route_index, 1);
  EXPECT_DOUBLE_EQ(drive.lanelets[1].s_start, 85.0);
  EXPECT_DOUBLE_EQ(drive.lanelets[1].s_end, 130.0);

  const auto on_b = project_to_drive_route(drive, 120, 120.0, 1.0, 5.0);
  ASSERT_TRUE(on_b.has_value());
  EXPECT_NEAR(on_b->s, 105.0, 1.0e-9);
  EXPECT_EQ(drive.route_index[on_b->index], 1);
  // At t=60 route A is active, and x=120 is not on the driven part of A.
  EXPECT_FALSE(project_to_drive_route(drive, 60, 120.0, 0.0, 5.0).has_value());
}

TEST(FrameGeometryTest, BuildDriveRouteSeparatesLapsOnTheSameRoad)
{
  // The same road is driven twice (lap 1 from t=0, lap 2 from t=100).
  const auto road = build_route_centerline({make_lanelet(1, {{0.0, 0.0}, {50.0, 0.0}})}, 1.0);
  const std::vector<RouteLegInput> legs = {{road, 0, "lap1"}, {road, 100, "lap2"}};
  std::vector<TimedPoint> samples;
  for (int i = 0; i <= 50; ++i) {
    samples.push_back({i, static_cast<double>(i), 0.0});
    samples.push_back({100 + i, static_cast<double>(i), 0.0});
  }
  const auto drive = build_drive_route(legs, samples, 5.0, 0.0);

  const auto lap1 = project_to_drive_route(drive, 10, 25.0, 0.0, 5.0);
  const auto lap2 = project_to_drive_route(drive, 110, 25.0, 0.0, 5.0);
  ASSERT_TRUE(lap1.has_value());
  ASSERT_TRUE(lap2.has_value());
  EXPECT_NEAR(lap1->s, 25.0, 1.0e-9);
  EXPECT_NEAR(lap2->s, 50.0 + 50.0 + 25.0, 1.0e-9);  // lap 1 + gap back to the start + 25
}

TEST(FrameGeometryTest, BuildDriveRouteSkipsRoutesThatWereNotDriven)
{
  std::vector<RouteLegInput> legs;
  legs.push_back(
    {build_route_centerline({make_lanelet(1, {{0.0, 0.0}, {50.0, 0.0}})}, 1.0), 0, "a"});
  legs.push_back(
    {build_route_centerline({make_lanelet(2, {{0.0, 100.0}, {50.0, 100.0}})}, 1.0), 100, "b"});
  const std::vector<TimedPoint> samples = {{10, 10.0, 0.0}, {20, 20.0, 0.0}};
  const auto drive = build_drive_route(legs, samples, 5.0, 0.0);
  EXPECT_TRUE(drive.routes[0].used);
  EXPECT_FALSE(drive.routes[1].used);
  EXPECT_FALSE(project_to_drive_route(drive, 150, 10.0, 100.0, 5.0).has_value());
}

}  // namespace autoware::planning_data_analyzer::utils
