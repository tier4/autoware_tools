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

#ifndef PERCEPTION_REPLAYER__ROSBAG_MANAGER_HPP_
#define PERCEPTION_REPLAYER__ROSBAG_MANAGER_HPP_

#include "type_alias.hpp"
#include "utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rcpputils/shared_library.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::planning_debug_tools
{

struct RosbagManagerParam
{
  bool enable_cache = false;
  double cache_window_before_sec = 30.0;
  double cache_window_after_sec = 60.0;
  std::string rosbag_format = "sqlite3";
};

class RosbagManager
{
public:
  struct RosbagInfo
  {
    std::string file_path;
    rclcpp::Time start_time;
    rclcpp::Time end_time;
  };

  explicit RosbagManager(rclcpp::Logger logger, bool tracked_object, const RosbagManagerParam & param);

  // Topic names
  std::string ego_odom_topic;
  std::string objects_topic;
  std::string traffic_signals_topic;
  std::string occupancy_grid_topic;
  std::string pointcloud_topic;
  std::string route_state_topic;
  std::string route_topic;

  // Initialize from rosbag path (handles both directory and single file)
  void initialize(const std::string & rosbag_path);

  void load_cache_data(const rclcpp::Time & base_timestamp);

  // Utility functions
  Odometry find_ego_odom_by_timestamp(const rclcpp::Time & timestamp) const;
  bool is_cache_initialized() const { return rosbag_cache_initialized_; }
  
  // Get rosbag time range
  rclcpp::Time get_bag_start_time() const;
  rclcpp::Time get_bag_end_timestamp() const;

private:
  rclcpp::Logger logger_;
  RosbagManagerParam param_;

  // Rosbag data storage
  std::vector<utils::DataStamped<Odometry>> rosbag_ego_odom_data_;
  std::vector<utils::DataStamped<PredictedObjects>> rosbag_predicted_objects_data_;
  std::vector<utils::DataStamped<TrackedObjects>> rosbag_tracked_objects_data_;
  std::vector<utils::DataStamped<TrafficLightGroupArray>> rosbag_traffic_signals_data_;
  std::vector<utils::DataStamped<OccupancyGrid>> rosbag_occupancy_grid_data_;
  std::vector<utils::DataStamped<PointCloud2>> rosbag_pointcloud_data_;
  std::vector<utils::DataStamped<RouteState>> rosbag_route_state_data_;
  std::vector<utils::DataStamped<LaneletRoute>> rosbag_route_data_;

  // Cache system
  std::vector<std::unique_ptr<rosbag2_cpp::Reader>> rosbag_readers_;
  rclcpp::Time current_cache_timestamp_;
  rclcpp::Time last_loaded_timestamp_;
  bool rosbag_cache_initialized_ = false;
  std::vector<RosbagInfo> cached_rosbag_infos_;

  // Type support for deserialization
  std::unordered_map<std::string, std::shared_ptr<const rosidl_message_type_support_t>>
    type_support_map_;
  std::unordered_map<std::string, std::shared_ptr<rcpputils::SharedLibrary>> type_support_libs_;

  // Helper functions
  void initialize_rosbag_cache(const std::vector<std::string> & rosbag_files);
  void load_all_ego_odom_data(const std::vector<std::string> & rosbag_files);
  void load_all_route_state_data(const std::vector<std::string> & rosbag_files);
  void load_type_support(const rosbag2_storage::TopicMetadata & topic_meta);
  void load_rosbag(const std::string & rosbag_path);
  void process_message(
    const rclcpp::Time & msg_timestamp, const std::string & topic_name,
    const std::shared_ptr<rcutils_uint8_array_t> & serialized_data);
};

}  // namespace autoware::planning_debug_tools

#endif  // PERCEPTION_REPLAYER__ROSBAG_MANAGER_HPP_

