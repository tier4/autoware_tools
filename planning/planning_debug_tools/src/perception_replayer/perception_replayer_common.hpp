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

#ifndef PERCEPTION_REPLAYER__PERCEPTION_REPLAYER_COMMON_HPP_
#define PERCEPTION_REPLAYER__PERCEPTION_REPLAYER_COMMON_HPP_

#include "type_alias.hpp"
#include "utils.hpp"

#include <autoware/object_recognition_utils/object_classification.hpp>
#include <autoware/universe_utils/ros/uuid_helper.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::planning_debug_tools
{

struct PerceptionReplayerCommonParam
{
  std::string rosbag_path;
  std::string rosbag_format;
  bool tracked_object;
  bool replay_route;
  std::vector<std::string> reference_image_topics;  // Topics for reference images

  // Distance-based object filter: objects farther than this from the current ego position are
  // not published. Disabled when <= 0.0.
  double object_filter_distance{0.0};
  // Semantic labels (autoware_perception_msgs::msg::ObjectClassification) the distance filter
  // applies to (AND condition with the distance check). Empty means all semantic types.
  std::vector<uint8_t> object_filter_semantics;
  // Object id prefixes (hex strings, as produced by autoware::universe_utils::toHexString; may
  // be the full 32-char id or just a leading prefix, e.g. the first 4 hex chars) whose matching
  // objects are never published, regardless of distance or semantic type.
  std::vector<std::string> object_filter_ids;
};

class PerceptionReplayerCommon : public rclcpp::Node
{
public:
  explicit PerceptionReplayerCommon(
    const PerceptionReplayerCommonParam & param, const std::string & node_name,
    const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());

  /**
   * @brief Get the rosbag start time
   * @return rclcpp::Time
   */
  rclcpp::Time get_bag_start_time() const
  {
    if (rosbag_ego_odom_data_.empty()) {
      throw std::runtime_error("No ego odom data available");
    }
    return rosbag_ego_odom_data_.front().first;
  }

  /**
   * @brief Get the rosbag end timestamp
   * @return rclcpp::Time
   */
  rclcpp::Time get_bag_end_timestamp() const
  {
    if (rosbag_ego_odom_data_.empty()) {
      throw std::runtime_error("No ego odom data available");
    }
    return rosbag_ego_odom_data_.back().first;
  }

  /**
   * @brief Publish objects and traffic light the given timestamp
   * @param bag_timestamp
   * @param current_timestamp
   * @param apply_noise Whether to apply perception noise to objects (default: false)
   */
  void publish_topics_at_timestamp(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp,
    const bool apply_noise = false);

  /**
   * @brief Publish the recorded ego pose for debugging
   * @param bag_timestamp
   */
  void publish_recorded_ego_pose(rclcpp::Time bag_timestamp);

  /**
   * @brief Publish the last recorded ego pose as goal pose
   */
  void publish_goal_pose();

  std::optional<Odometry> get_latest_ego_odom() const
  {
    return ego_odom_ ? std::make_optional<Odometry>(*ego_odom_) : std::nullopt;
  }

  /**
   * @brief Publish traffic light the given timestamp
   * @param bag_timestamp
   * @param current_timestamp
   */
  void publish_traffic_lights_at_timestamp(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp);

  /**
   * @brief Publish reference images at the given timestamp
   * @param bag_timestamp
   * @param current_timestamp
   */
  void publish_reference_images_at_timestamp(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp);

  /**
   * @brief Publish route and route state (transient_local) at the given timestamp.
   * Only publishes when the message to publish differs from the last published one.
   * @param bag_timestamp Used to select messages from the rosbag.
   * @param current_timestamp Stamp written on published messages (replay clock).
   */
  void publish_route_at_timestamp(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp);

protected:
  const PerceptionReplayerCommonParam param_;

  // Template function to apply perception noise to objects
  template <typename T>
  void apply_perception_noise(
    T & msg, const double update_rate = 0.03, const double x_noise_std = 0.1,
    const double y_noise_std = 0.05)
  {
    if (uniform_dist_(gen_) < update_rate) {
      noise_cache_.clear();
    }

    for (auto & object : msg.objects) {
      const UUID & object_uuid = object.object_id;

      auto it = noise_cache_.find(object_uuid);
      if (it == noise_cache_.end()) {
        it = noise_cache_
               .emplace(
                 object_uuid,
                 std::make_pair(
                   standard_dist_(gen_) * x_noise_std, standard_dist_(gen_) * y_noise_std))
               .first;
      }
      const double noise_x = it->second.first;
      const double noise_y = it->second.second;

      geometry_msgs::msg::Pose & pose = [&]() -> geometry_msgs::msg::Pose & {
        if constexpr (std::is_same_v<T, PredictedObjects>) {
          return object.kinematics.initial_pose_with_covariance.pose;
        } else {
          return object.kinematics.pose_with_covariance.pose;
        }
      }();

      const double obj_yaw = utils::get_yaw_from_quaternion(pose.orientation);
      const double noise_x_world = noise_x * std::cos(obj_yaw) - noise_y * std::sin(obj_yaw);
      const double noise_y_world = noise_x * std::sin(obj_yaw) + noise_y * std::cos(obj_yaw);
      pose.position.x += noise_x_world;
      pose.position.y += noise_y_world;
    }
  }

  // Drop objects that should not be published:
  //  - any object whose id (hex string) starts with one of param_.object_filter_ids,
  //    unconditionally. Entries may be a full 32-char id or just a leading prefix (e.g. the
  //    first 4 hex chars); and
  //  - objects farther than param_.object_filter_distance from the current ego position, if that
  //    filter is enabled (> 0.0). When param_.object_filter_semantics is non-empty, only objects
  //    whose highest-probability classification is in that list are subject to the distance
  //    filter (AND condition); otherwise the distance filter applies to every object.
  template <typename T>
  void filter_objects(T & msg) const
  {
    const bool id_filter_enabled = !param_.object_filter_ids.empty();
    const bool distance_filter_enabled = param_.object_filter_distance > 0.0 && ego_odom_;
    if (!id_filter_enabled && !distance_filter_enabled) {
      return;
    }

    const auto is_filtered_id = [this](const auto & object) {
      const auto id_hex = autoware::universe_utils::toHexString(object.object_id);
      return std::any_of(
        param_.object_filter_ids.begin(), param_.object_filter_ids.end(),
        [&id_hex](const std::string & prefix) { return id_hex.rfind(prefix, 0) == 0; });
    };

    const auto is_subject_to_distance_filter = [this](const auto & object) {
      if (param_.object_filter_semantics.empty()) {
        return true;
      }
      const auto label = autoware::object_recognition_utils::getHighestProbLabel(
        object.classification);
      return std::find(
               param_.object_filter_semantics.begin(), param_.object_filter_semantics.end(),
               label) != param_.object_filter_semantics.end();
    };

    const double filter_distance_squared =
      param_.object_filter_distance * param_.object_filter_distance;
    const auto ego_position =
      distance_filter_enabled ? ego_odom_->pose.pose.position : geometry_msgs::msg::Point{};

    auto & objects = msg.objects;
    objects.erase(
      std::remove_if(
        objects.begin(), objects.end(),
        [&](const auto & object) {
          if (id_filter_enabled && is_filtered_id(object)) {
            return true;
          }

          if (!distance_filter_enabled || !is_subject_to_distance_filter(object)) {
            return false;
          }

          const geometry_msgs::msg::Point & object_position = [&]() -> const
            geometry_msgs::msg::Point &
          {
            if constexpr (std::is_same_v<T, PredictedObjects>) {
              return object.kinematics.initial_pose_with_covariance.pose.position;
            } else {
              return object.kinematics.pose_with_covariance.pose.position;
            }
          }();

          const double dx = object_position.x - ego_position.x;
          const double dy = object_position.y - ego_position.y;
          return (dx * dx + dy * dy) > filter_distance_squared;
        }),
      objects.end());
  }

  // noise
  mutable std::random_device rd_;
  mutable std::mt19937 gen_;
  mutable std::uniform_real_distribution<double> uniform_dist_;
  mutable std::normal_distribution<double> standard_dist_;

  struct UUIDHash
  {
    std::size_t operator()(const UUID & uuid) const
    {
      return *reinterpret_cast<const std::size_t *>(uuid.uuid.data());
    }
  };
  struct UUIDEqual
  {
    bool operator()(const UUID & lhs, const UUID & rhs) const { return lhs.uuid == rhs.uuid; }
  };
  mutable std::unordered_map<UUID, std::pair<double, double>, UUIDHash, UUIDEqual> noise_cache_;

  // rosbag data
  std::vector<utils::DataStamped<Odometry>> rosbag_ego_odom_data_;
  std::vector<utils::DataStamped<PredictedObjects>> rosbag_predicted_objects_data_;
  std::vector<utils::DataStamped<TrackedObjects>> rosbag_tracked_objects_data_;
  std::vector<utils::DataStamped<TrafficLightGroupArray>> rosbag_traffic_signals_data_;
  std::vector<utils::DataStamped<OccupancyGrid>> rosbag_occupancy_grid_data_;
  std::vector<utils::DataStamped<LaneletRoute>> rosbag_route_data_;
  std::vector<utils::DataStamped<RouteState>> rosbag_route_state_data_;

  // Reference image data: topic name -> timestamped messages
  std::unordered_map<std::string, std::vector<utils::DataStamped<CompressedImage>>>
    rosbag_reference_image_data_;

  // load rosbag
  void load_rosbag(const std::string & rosbag_path, const std::string & rosbag_format);
  std::vector<std::string> find_rosbag_files(
    const std::string & directory_path, const std::string & rosbag_format) const;
  Odometry find_ego_odom_by_timestamp(const rclcpp::Time & timestamp) const;

  void kill_online_perception_node();
  void kill_process(const std::string & process_name);
  void unload_component(const std::string & container_name, const std::string & component_name);

  // subscriber
  void on_ego_odom(const Odometry::SharedPtr msg);
  rclcpp::Subscription<Odometry>::SharedPtr ego_odom_sub_;
  Odometry::SharedPtr ego_odom_;

  // timers
  rclcpp::CallbackGroup::SharedPtr callback_group_check_perception_;
  rclcpp::TimerBase::SharedPtr timer_check_perception_process_;

  // publisher
  rclcpp::PublisherBase::SharedPtr objects_pub_;
  rclcpp::Publisher<TrafficLightGroupArray>::SharedPtr traffic_signals_pub_;
  rclcpp::Publisher<OccupancyGrid>::SharedPtr occupancy_grid_pub_;
  rclcpp::Publisher<LaneletRoute>::SharedPtr route_pub_;
  rclcpp::Publisher<RouteState>::SharedPtr route_state_pub_;

  // track last published index to avoid redundant publishes for transient_local topics
  std::optional<size_t> last_published_route_idx_;
  std::optional<size_t> last_published_route_state_idx_;
  /// Last time route_state was published (replay clock); used for periodic re-publish.
  std::optional<rclcpp::Time> last_route_state_publish_replay_time_;

  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr recorded_ego_as_initialpose_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr goal_as_mission_planning_goal_pub_;

  rclcpp::Publisher<Odometry>::SharedPtr recorded_ego_pub_;

  // Reference image publishers: topic name -> publisher
  std::unordered_map<std::string, rclcpp::Publisher<CompressedImage>::SharedPtr>
    reference_image_pubs_;
};

}  // namespace autoware::planning_debug_tools

#endif  // PERCEPTION_REPLAYER__PERCEPTION_REPLAYER_COMMON_HPP_
