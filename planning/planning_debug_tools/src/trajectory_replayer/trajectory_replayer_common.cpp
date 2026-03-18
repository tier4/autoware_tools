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

#include "trajectory_replayer_common.hpp"

#include "../perception_replayer/serialized_bag_message.hpp"
#include "../perception_replayer/utils.hpp"

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::planning_debug_tools::trajectory_replayer
{

namespace
{
constexpr auto odom_topic = "/localization/kinematic_state";
constexpr auto accel_topic = "/localization/acceleration";
constexpr auto objects_topic = "/perception/object_recognition/objects";
constexpr auto pointcloud_topic = "/perception/obstacle_segmentation/pointcloud";
constexpr auto diffusion_output_topic =
  "/planning/generator/diffusion_planner/candidate_trajectories";
constexpr auto modifier_output_topic =
  "/planning/generator/diffusion_planner/modified_candidate_trajectories";
constexpr auto optimizer_output_topic = "/planning/generator/candidate_trajectories";
constexpr auto original_optimizer_output_topic =
  "/trajectory_replayer/original/candidate_trajectories";
constexpr auto tf_topic = "/tf";
constexpr auto tf_static_topic = "/tf_static";
}  // namespace

std::vector<std::string> TrajectoryReplayerCommon::find_rosbag_files(
  const std::string & directory_path, const std::string & rosbag_format) const
{
  const std::string extension = (rosbag_format == "mcap") ? ".mcap" : ".db3";
  std::vector<std::string> rosbag_files;

  for (const auto & entry : std::filesystem::directory_iterator(directory_path)) {
    if (entry.is_regular_file() && entry.path().extension() == extension) {
      rosbag_files.push_back(entry.path().string());
    }
  }

  std::sort(rosbag_files.begin(), rosbag_files.end());
  return rosbag_files;
}

void TrajectoryReplayerCommon::load_rosbag(
  const std::string & rosbag_path, const std::string & rosbag_format)
{
  std::cout << "Loading rosbag: " << rosbag_path << std::endl;

  auto reader = std::make_unique<rosbag2_cpp::Reader>();

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = rosbag_path;
  storage_options.storage_id = rosbag_format;

  reader->open(storage_options);

  const auto topics = reader->get_all_topics_and_types();
  std::cout << "Found " << topics.size() << " topics in bag" << std::endl;

  std::unordered_map<std::string, std::shared_ptr<const rosidl_message_type_support_t>>
    type_support_map;
  std::unordered_map<std::string, std::shared_ptr<rcpputils::SharedLibrary>> type_support_libs;

  for (const auto & topic_meta : topics) {
    try {
      auto library =
        rosbag2_cpp::get_typesupport_library(topic_meta.type, "rosidl_typesupport_cpp");
      type_support_libs[topic_meta.name] = library;

      const rosidl_message_type_support_t * type_support =
        rosbag2_cpp::get_typesupport_handle(topic_meta.type, "rosidl_typesupport_cpp", library);

      if (type_support) {
        type_support_map[topic_meta.name] = std::shared_ptr<const rosidl_message_type_support_t>(
          type_support, [](const rosidl_message_type_support_t *) {});
      }
    } catch (const std::exception & e) {
      std::cerr << "Warning: Could not load type support for topic " << topic_meta.name
                << std::endl;
    }
  }

  rosbag2_storage::StorageFilter storage_filter;
  storage_filter.topics = {odom_topic, accel_topic, optimizer_output_topic, objects_topic,
                           tf_topic, tf_static_topic};

  if (param_.mode == ReplayMode::MODIFIER) {
    storage_filter.topics.push_back(diffusion_output_topic);
    if (param_.replay_pointcloud) {
      storage_filter.topics.push_back(pointcloud_topic);
    }
  } else {
    storage_filter.topics.push_back(modifier_output_topic);
  }

  reader->set_filter(storage_filter);

  while (reader->has_next()) {
    try {
      auto bag_message = reader->read_next();

      auto it = type_support_map.find(bag_message->topic_name);
      if (it == type_support_map.end() || !it->second) {
        continue;
      }

      const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));

      if (bag_message->topic_name == odom_topic) {
        auto msg = utils::deserialize_message<Odometry>(bag_message->serialized_data);
        rosbag_odom_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == accel_topic) {
        auto msg =
          utils::deserialize_message<AccelWithCovarianceStamped>(bag_message->serialized_data);
        rosbag_accel_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == diffusion_output_topic) {
        auto msg =
          utils::deserialize_message<CandidateTrajectories>(bag_message->serialized_data);
        rosbag_diffusion_output_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == modifier_output_topic) {
        auto msg =
          utils::deserialize_message<CandidateTrajectories>(bag_message->serialized_data);
        rosbag_modifier_output_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == optimizer_output_topic) {
        auto msg =
          utils::deserialize_message<CandidateTrajectories>(bag_message->serialized_data);
        rosbag_original_optimizer_output_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == objects_topic) {
        auto msg = utils::deserialize_message<PredictedObjects>(bag_message->serialized_data);
        rosbag_objects_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == pointcloud_topic) {
        auto msg = utils::deserialize_message<PointCloud2>(bag_message->serialized_data);
        rosbag_pointcloud_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == tf_topic) {
        auto msg = utils::deserialize_message<TFMessage>(bag_message->serialized_data);
        rosbag_tf_data_.emplace_back(timestamp, *msg);
      } else if (bag_message->topic_name == tf_static_topic) {
        auto msg = utils::deserialize_message<TFMessage>(bag_message->serialized_data);
        rosbag_tf_static_data_.push_back(*msg);
      }
    } catch (const std::exception & e) {
      std::cerr << "Error reading message: " << e.what() << std::endl;
      continue;
    }
  }

  std::cout << "Finished loading rosbag: " << rosbag_path << std::endl;
}

TrajectoryReplayerCommon::TrajectoryReplayerCommon(
  const TrajectoryReplayerParam & param, const std::string & node_name,
  const rclcpp::NodeOptions & node_options)
: Node(node_name, node_options), param_(param)
{
  if (std::filesystem::is_directory(param_.rosbag_path)) {
    std::cout << "Processing rosbag directory: " << param_.rosbag_path << std::endl;
    const auto rosbag_files = find_rosbag_files(param_.rosbag_path, param_.rosbag_format);
    std::cout << "Found " << rosbag_files.size() << " bag files" << std::endl;
    for (size_t i = 0; i < rosbag_files.size(); ++i) {
      std::cout << "Loading bag file " << (i + 1) << "/" << rosbag_files.size() << ": "
                << rosbag_files[i] << std::endl;
      load_rosbag(rosbag_files[i], param_.rosbag_format);
    }
  } else {
    std::cout << "Loading single bag file: " << param_.rosbag_path << std::endl;
    load_rosbag(param_.rosbag_path, param_.rosbag_format);
  }

  std::cout << "Loaded data sizes:" << std::endl;
  std::cout << "  odom: " << rosbag_odom_data_.size() << std::endl;
  std::cout << "  accel: " << rosbag_accel_data_.size() << std::endl;
  std::cout << "  diffusion output: " << rosbag_diffusion_output_data_.size() << std::endl;
  std::cout << "  modifier output: " << rosbag_modifier_output_data_.size() << std::endl;
  std::cout << "  optimizer output (original): " << rosbag_original_optimizer_output_data_.size()
            << std::endl;
  std::cout << "  objects: " << rosbag_objects_data_.size() << std::endl;
  std::cout << "  pointcloud: " << rosbag_pointcloud_data_.size() << std::endl;
  std::cout << "  tf: " << rosbag_tf_data_.size() << std::endl;
  std::cout << "  tf_static: " << rosbag_tf_static_data_.size() << std::endl;

  // input publishers: feed live nodes
  if (param_.mode == ReplayMode::MODIFIER) {
    trajectories_input_pub_ =
      this->create_publisher<CandidateTrajectories>(diffusion_output_topic, 1);
    if (param_.replay_pointcloud) {
      pointcloud_pub_ = this->create_publisher<PointCloud2>(pointcloud_topic, 1);
    }
  } else {
    trajectories_input_pub_ =
      this->create_publisher<CandidateTrajectories>(modifier_output_topic, 1);
  }

  objects_pub_ = this->create_publisher<PredictedObjects>(objects_topic, 1);

  odom_pub_ = this->create_publisher<Odometry>(odom_topic, 1);
  accel_pub_ = this->create_publisher<AccelWithCovarianceStamped>(accel_topic, 1);

  // TF publishers
  tf_pub_ = this->create_publisher<TFMessage>(tf_topic, 100);
  rclcpp::QoS tf_static_qos(100);
  tf_static_qos.transient_local();
  tf_static_pub_ = this->create_publisher<TFMessage>(tf_static_topic, tf_static_qos);

  // original output publisher (for comparison)
  original_optimizer_output_pub_ =
    this->create_publisher<CandidateTrajectories>(original_optimizer_output_topic, 1);

  callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  publish_tf_static();

  RCLCPP_INFO(
    get_logger(), "Trajectory replayer ready (mode=%s)",
    param_.mode == ReplayMode::MODIFIER ? "modifier" : "optimizer");
}

rclcpp::Time TrajectoryReplayerCommon::get_bag_start_time() const
{
  if (rosbag_odom_data_.empty()) {
    throw std::runtime_error("No odometry data available");
  }
  return rosbag_odom_data_.front().first;
}

rclcpp::Time TrajectoryReplayerCommon::get_bag_end_timestamp() const
{
  if (rosbag_odom_data_.empty()) {
    throw std::runtime_error("No odometry data available");
  }
  return rosbag_odom_data_.back().first;
}

void TrajectoryReplayerCommon::publish_tf_static()
{
  for (const auto & msg : rosbag_tf_static_data_) {
    tf_static_pub_->publish(msg);
  }
  if (!rosbag_tf_static_data_.empty()) {
    RCLCPP_INFO(get_logger(), "Published %zu tf_static messages", rosbag_tf_static_data_.size());
  }
}

void TrajectoryReplayerCommon::publish_sensor_data(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  // publish odometry
  if (!rosbag_odom_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_odom_data_, bag_timestamp);
    auto msg = rosbag_odom_data_[idx].second;
    msg.header.stamp = current_timestamp;
    odom_pub_->publish(msg);
  }

  // publish acceleration
  if (!rosbag_accel_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_accel_data_, bag_timestamp);
    auto msg = rosbag_accel_data_[idx].second;
    msg.header.stamp = current_timestamp;
    accel_pub_->publish(msg);
  }

  // publish objects (both modes, for BEV visualization)
  if (!rosbag_objects_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_objects_data_, bag_timestamp);
    auto msg = rosbag_objects_data_[idx].second;
    msg.header.stamp = current_timestamp;
    objects_pub_->publish(msg);
  }

  // publish pointcloud (modifier mode only)
  if (param_.replay_pointcloud && !rosbag_pointcloud_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_pointcloud_data_, bag_timestamp);
    auto msg = rosbag_pointcloud_data_[idx].second;
    msg.header.stamp = current_timestamp;
    pointcloud_pub_->publish(msg);
  }

  // publish TF
  if (!rosbag_tf_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_tf_data_, bag_timestamp);
    auto msg = rosbag_tf_data_[idx].second;
    for (auto & transform : msg.transforms) {
      transform.header.stamp = current_timestamp;
    }
    tf_pub_->publish(msg);
  }
}

void TrajectoryReplayerCommon::publish_trajectory_input(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  // publish trajectory input to live nodes
  if (param_.mode == ReplayMode::MODIFIER) {
    if (!rosbag_diffusion_output_data_.empty()) {
      const size_t idx =
        utils::get_nearest_index(rosbag_diffusion_output_data_, bag_timestamp);
      auto msg = rosbag_diffusion_output_data_[idx].second;
      for (auto & traj : msg.candidate_trajectories) {
        traj.header.stamp = current_timestamp;
      }
      trajectories_input_pub_->publish(msg);
    }
  } else {
    if (!rosbag_modifier_output_data_.empty()) {
      const size_t idx =
        utils::get_nearest_index(rosbag_modifier_output_data_, bag_timestamp);
      auto msg = rosbag_modifier_output_data_[idx].second;
      for (auto & traj : msg.candidate_trajectories) {
        traj.header.stamp = current_timestamp;
      }
      trajectories_input_pub_->publish(msg);
    }
  }

  // publish original optimizer output for comparison
  if (!rosbag_original_optimizer_output_data_.empty()) {
    const size_t idx =
      utils::get_nearest_index(rosbag_original_optimizer_output_data_, bag_timestamp);
    auto msg = rosbag_original_optimizer_output_data_[idx].second;
    for (auto & traj : msg.candidate_trajectories) {
      traj.header.stamp = current_timestamp;
    }
    original_optimizer_output_pub_->publish(msg);
  }
}

}  // namespace autoware::planning_debug_tools::trajectory_replayer
