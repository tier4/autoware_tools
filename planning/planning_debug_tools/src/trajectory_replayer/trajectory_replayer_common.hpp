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

#ifndef TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_COMMON_HPP_
#define TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_COMMON_HPP_

#include "../perception_replayer/utils.hpp"
#include "type_alias.hpp"

#include <rclcpp/rclcpp.hpp>

#include <string>
#include <vector>

namespace autoware::planning_debug_tools::trajectory_replayer
{

enum class ReplayMode { MODIFIER, OPTIMIZER };

struct TrajectoryReplayerParam
{
  std::string rosbag_path;
  std::string rosbag_format;
  ReplayMode mode;
  bool replay_pointcloud;
};

class TrajectoryReplayerCommon : public rclcpp::Node
{
public:
  explicit TrajectoryReplayerCommon(
    const TrajectoryReplayerParam & param, const std::string & node_name,
    const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());

  rclcpp::Time get_bag_start_time() const;
  rclcpp::Time get_bag_end_timestamp() const;

  void publish_sensor_data(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp);
  void publish_trajectory_input(
    const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp);

protected:
  const TrajectoryReplayerParam param_;

  // rosbag data: inputs shared by modifier and optimizer
  std::vector<utils::DataStamped<Odometry>> rosbag_odom_data_;
  std::vector<utils::DataStamped<AccelWithCovarianceStamped>> rosbag_accel_data_;

  // modifier-mode inputs (diffusion planner output + perception)
  std::vector<utils::DataStamped<CandidateTrajectories>> rosbag_diffusion_output_data_;
  std::vector<utils::DataStamped<PredictedObjects>> rosbag_objects_data_;
  std::vector<utils::DataStamped<PointCloud2>> rosbag_pointcloud_data_;

  // optimizer-mode input (modifier output)
  std::vector<utils::DataStamped<CandidateTrajectories>> rosbag_modifier_output_data_;

  // original outputs from bag (for comparison)
  std::vector<utils::DataStamped<CandidateTrajectories>> rosbag_original_optimizer_output_data_;

  // TF data
  std::vector<utils::DataStamped<TFMessage>> rosbag_tf_data_;
  std::vector<TFMessage> rosbag_tf_static_data_;

  rclcpp::CallbackGroup::SharedPtr callback_group_;

private:
  void load_rosbag(const std::string & rosbag_path, const std::string & rosbag_format);
  std::vector<std::string> find_rosbag_files(
    const std::string & directory_path, const std::string & rosbag_format) const;
  void publish_tf_static();

  // publishers: replay inputs to live nodes
  rclcpp::Publisher<CandidateTrajectories>::SharedPtr trajectories_input_pub_;
  rclcpp::Publisher<Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<AccelWithCovarianceStamped>::SharedPtr accel_pub_;
  rclcpp::Publisher<PredictedObjects>::SharedPtr objects_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pointcloud_pub_;

  // publishers: TF
  rclcpp::Publisher<TFMessage>::SharedPtr tf_pub_;
  rclcpp::Publisher<TFMessage>::SharedPtr tf_static_pub_;

  // publishers: original outputs for comparison
  rclcpp::Publisher<CandidateTrajectories>::SharedPtr original_optimizer_output_pub_;
};

}  // namespace autoware::planning_debug_tools::trajectory_replayer

#endif  // TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_COMMON_HPP_
