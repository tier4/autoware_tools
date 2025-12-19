// Copyright 2025 TIER IV, Inc.
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

#include "perception_replayer_common.hpp"

#include "utils.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::planning_debug_tools
{

PerceptionReplayerCommon::PerceptionReplayerCommon(
  const PerceptionReplayerCommonParam & param, std::unique_ptr<RosbagManager> rosbag_manager,
  const std::string & node_name, const rclcpp::NodeOptions & node_options)
: Node(node_name, node_options), param_(param), rosbag_manager_(std::move(rosbag_manager))
{

  // Subscriber
  ego_odom_sub_ = this->create_subscription<Odometry>(
    rosbag_manager_->get_ego_odom_topic(), 1,
    std::bind(&PerceptionReplayerCommon::on_ego_odom, this, std::placeholders::_1));

  // Publisher
  recorded_ego_pub_ = this->create_publisher<Odometry>("/perception_reproducer/rosbag_ego_odom", 1);

  // create objects publisher based on the option
  if (param_.tracked_object) {
    objects_pub_ =
      this->create_publisher<TrackedObjects>("/perception/object_recognition/tracking/objects", 1);
  } else {
    objects_pub_ =
      this->create_publisher<PredictedObjects>("/perception/object_recognition/objects", 1);
  }

  traffic_signals_pub_ = this->create_publisher<TrafficLightGroupArray>(
    "/perception/traffic_light_recognition/traffic_signals", 1);

  rclcpp::QoS occupancy_grid_qos(1);
  occupancy_grid_qos.transient_local();
  occupancy_grid_pub_ =
    this->create_publisher<OccupancyGrid>("/perception/occupancy_grid_map/map", occupancy_grid_qos);

  rclcpp::QoS pointcloud_qos(1);
  pointcloud_qos.best_effort();
  pointcloud_pub_ = this->create_publisher<PointCloud2>(
    "/perception/obstacle_segmentation/pointcloud", pointcloud_qos);

  recorded_ego_as_initialpose_pub_ =
    this->create_publisher<PoseWithCovarianceStamped>("/initialpose", 1);
  goal_as_mission_planning_goal_pub_ =
    this->create_publisher<PoseStamped>("/planning/mission_planning/goal", 1);

  // route publishers with transient_local QoS
  rclcpp::QoS route_qos(1);
  route_qos.transient_local();
  route_state_pub_ = this->create_publisher<RouteState>(
    "/planning/mission_planning/state", route_qos);
  route_pub_ = this->create_publisher<LaneletRoute>(
    "/planning/mission_planning/route", route_qos);

  // kill online perception nodes once at initialization
  kill_online_perception_node();
}

void PerceptionReplayerCommon::publish_topics_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  // for debugging
  recorded_ego_pub_->publish(rosbag_manager_->find_ego_odom_by_timestamp(bag_timestamp));

  // publish objects
  if (param_.tracked_object) {
    const auto & tracked_objects_data = rosbag_manager_->get_tracked_objects_data();
    const auto objects_msg =
      utils::find_message_by_timestamp(tracked_objects_data, bag_timestamp);
    if (objects_msg.has_value()) {
      auto msg = objects_msg.value();
      msg.header.stamp = current_timestamp;
      auto publisher = std::dynamic_pointer_cast<rclcpp::Publisher<TrackedObjects>>(objects_pub_);
      if (publisher) {
        publisher->publish(msg);
      }
    }
  } else {
    const auto & predicted_objects_data = rosbag_manager_->get_predicted_objects_data();
    const auto objects_msg =
      utils::find_message_by_timestamp(predicted_objects_data, bag_timestamp);
    if (objects_msg.has_value()) {
      auto msg = objects_msg.value();
      msg.header.stamp = current_timestamp;
      auto publisher = std::dynamic_pointer_cast<rclcpp::Publisher<PredictedObjects>>(objects_pub_);
      if (publisher) {
        publisher->publish(msg);
      }
    }
  }

  publish_traffic_lights_at_timestamp(bag_timestamp, current_timestamp);

  // publish occupancy grid
  const auto & occupancy_grid_data = rosbag_manager_->get_occupancy_grid_data();
  if (!occupancy_grid_data.empty()) {
    const size_t idx = utils::get_nearest_index(occupancy_grid_data, bag_timestamp);
    auto & msg = occupancy_grid_data[idx].second;
    msg.header.stamp = current_timestamp;
    occupancy_grid_pub_->publish(msg);
  }

  // publish pointcloud with coordinate conversion
  const auto & pointcloud_data = rosbag_manager_->get_pointcloud_data();
  if (!pointcloud_data.empty()) {
    const auto ego_odom = get_latest_ego_odom();
    if (ego_odom.has_value()) {
      const auto ego_pose = ego_odom->pose.pose;
      const auto log_ego_pose = rosbag_manager_->find_ego_odom_by_timestamp(bag_timestamp).pose.pose;

      const size_t idx = utils::get_nearest_index(pointcloud_data, bag_timestamp);
      auto msg = pointcloud_data[idx].second;
      utils::translate_pointcloud_coordinate(ego_pose, log_ego_pose, msg);
      msg.header.stamp = current_timestamp;
      pointcloud_pub_->publish(std::move(msg));
    }
  }
}

void PerceptionReplayerCommon::publish_traffic_lights_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  const auto & traffic_signals_data = rosbag_manager_->get_traffic_signals_data();
  const auto traffic_signals_msg =
    utils::find_message_by_timestamp(traffic_signals_data, bag_timestamp);
  if (traffic_signals_msg.has_value()) {
    auto msg = traffic_signals_msg.value();
    const auto original_timestamp = msg.stamp;
    msg.stamp = current_timestamp;

    for (auto & traffic_signals_group : msg.traffic_light_groups) {
      for (auto & prediction : traffic_signals_group.predictions) {
        // time difference between original stamp and prediction stamp
        const auto time_diff = rclcpp::Time(prediction.predicted_stamp) - original_timestamp;

        // fix timestamp
        prediction.predicted_stamp = current_timestamp + time_diff;
      }
    }

    traffic_signals_pub_->publish(msg);
  }
}

void PerceptionReplayerCommon::publish_recorded_ego_pose(rclcpp::Time bag_timestamp)
{
  const auto ego_odom = rosbag_manager_->find_ego_odom_by_timestamp(bag_timestamp);

  PoseWithCovarianceStamped initialpose;
  initialpose.header.stamp = this->get_clock()->now();
  initialpose.header.frame_id = "map";
  initialpose.pose.pose = ego_odom.pose.pose;

  // clang-format off
  initialpose.pose.covariance = {
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.06853892326654787,
  };
  // clang-format on

  recorded_ego_as_initialpose_pub_->publish(initialpose);

  RCLCPP_INFO(get_logger(), "Published recorded ego pose as /initialpose");
}

void PerceptionReplayerCommon::publish_goal_pose()
{
  const auto ego_pose = rosbag_manager_->find_ego_odom_by_timestamp(rosbag_manager_->get_bag_end_timestamp());

  PoseStamped goal_pose;
  goal_pose.header.stamp = this->get_clock()->now();
  goal_pose.header.frame_id = "map";
  goal_pose.pose = ego_pose.pose.pose;

  goal_as_mission_planning_goal_pub_->publish(goal_pose);
  RCLCPP_INFO(get_logger(), "Published last recorded ego pose as /planning/mission_planning/goal");
}

void PerceptionReplayerCommon::on_ego_odom(const Odometry::SharedPtr msg)
{
  ego_odom_ = msg;
}

void PerceptionReplayerCommon::kill_online_perception_node()
{
  // kill the object recognition node
  if (param_.tracked_object && !rosbag_manager_->get_tracked_objects_data().empty()) {
    kill_process("multi_object_tracker");
  } else if (!rosbag_manager_->get_predicted_objects_data().empty()) {
    kill_process("map_based_prediction");
  }

  // unload the occupancy grid map node
  if (!rosbag_manager_->get_occupancy_grid_data().empty()) {
    unload_component("/pointcloud_container", "occupancy_grid_map_node");
  }

  // kill dummy_perception_publisher
  if (!rosbag_manager_->get_pointcloud_data().empty()) {
    kill_process("dummy_perception_publisher");
  }
}

void PerceptionReplayerCommon::kill_process(const std::string & process_name)
{
  // use pidof to find the process
  const std::string command = "pidof " + process_name + " 2>/dev/null";
  FILE * pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return;
  }

  char buffer[128];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  const int pclose_result = pclose(pipe);

  // check if pidof found the process (exit code 0)
  if (pclose_result == 0 && !result.empty()) {
    try {
      // parse pid from result
      const pid_t pid = static_cast<pid_t>(std::stol(result));
      // send SIGTERM (same as Python's process.terminate())
      const std::string kill_command = "kill -TERM " + std::to_string(pid) + " 2>/dev/null";
      const int kill_result = system(kill_command.c_str());
      if (kill_result == 0) {
        RCLCPP_INFO(get_logger(), "Terminated process %s (PID: %d)", process_name.c_str(), pid);
      }
    } catch (const std::exception & e) {
      // failed to convert pid, ignore
    }
  }
}

void PerceptionReplayerCommon::unload_component(
  const std::string & container_name, const std::string & component_name)
{
  const std::string list_base = "ros2 component list " + container_name + " 2>/dev/null | ";
  const std::string grep_component = "grep " + component_name;

  const std::string check_command = list_base + "grep -q " + component_name + " 2>/dev/null";
  if (system(check_command.c_str()) != 0) {
    return;
  }

  const std::string unload_command = list_base + grep_component +
                                     " | awk '{print $1}' | "
                                     "xargs -I {} ros2 component unload " +
                                     container_name + " {} 2>/dev/null || true";
  const int unload_result = system(unload_command.c_str());
  (void)unload_result;
}

void PerceptionReplayerCommon::check_and_publish_route(
  const rclcpp::Time & current_ego_odom_timestamp)
{
  const auto & route_state_data = rosbag_manager_->get_route_state_data();
  const auto & route_data = rosbag_manager_->get_route_data();

  // Publish route_state if timestamp is reached
  while (next_route_state_idx_ < route_state_data.size()) {
    const auto & route_state = route_state_data[next_route_state_idx_];
    if (route_state.first <= current_ego_odom_timestamp) {
      auto msg = route_state.second;
      msg.stamp = this->get_clock()->now();
      route_state_pub_->publish(msg);
      ++next_route_state_idx_;
      RCLCPP_INFO(
        get_logger(), "Published route_state with original timestamp: %f",
        route_state.first.seconds());
    } else {
      break;
    }
  }

  // Publish route if timestamp is reached
  while (next_route_idx_ < route_data.size()) {
    const auto & route = route_data[next_route_idx_];
    if (route.first <= current_ego_odom_timestamp) {
      auto msg = route.second;
      msg.header.stamp = this->get_clock()->now();
      route_pub_->publish(msg);
      ++next_route_idx_;
    } else {
      break;
    }
  }
}

void PerceptionReplayerCommon::reset_route_cache()
{
  next_route_state_idx_ = 0;
  next_route_idx_ = 0;
}

void PerceptionReplayerCommon::load_cache_data(const rclcpp::Time & bag_timestamp)
{
  if (rosbag_manager_ && rosbag_manager_->is_cache_initialized() && rosbag_manager_->is_cache_enabled()) {
    rosbag_manager_->load_cache_data(bag_timestamp);
  }
}

}  // namespace autoware::planning_debug_tools
