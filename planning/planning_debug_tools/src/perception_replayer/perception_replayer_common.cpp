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

#include "serialized_bag_message.hpp"
#include "utils.hpp"

#include <rclcpp/typesupport_helpers.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/metadata_io.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

int compare_natural(const std::string & a, const std::string & b)
{
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() && j < b.size()) {
    const auto is_digit_a = std::isdigit(static_cast<unsigned char>(a[i])) != 0;
    const auto is_digit_b = std::isdigit(static_cast<unsigned char>(b[j])) != 0;
    if (is_digit_a && is_digit_b) {
      while (i < a.size() && a[i] == '0') {
        ++i;
      }
      while (j < b.size() && b[j] == '0') {
        ++j;
      }

      const size_t start_i = i;
      const size_t start_j = j;

      while (i < a.size() && std::isdigit(static_cast<unsigned char>(a[i])) != 0) {
        ++i;
      }
      while (j < b.size() && std::isdigit(static_cast<unsigned char>(b[j])) != 0) {
        ++j;
      }

      const size_t len_a = i - start_i;
      const size_t len_b = j - start_j;
      if (len_a != len_b) {
        return (len_a < len_b) ? -1 : 1;
      }

      const int cmp = a.compare(start_i, len_a, b, start_j, len_b);
      if (cmp != 0) {
        return cmp;
      }
      continue;
    }

    if (a[i] != b[j]) {
      return (a[i] < b[j]) ? -1 : 1;
    }
    ++i;
    ++j;
  }

  if (i < a.size()) {
    return 1;
  }
  if (j < b.size()) {
    return -1;
  }
  return 0;
}

bool natural_path_less(const std::string & a, const std::string & b)
{
  const auto basename = [](const std::string & path) {
    return std::filesystem::path(path).filename().string();
  };
  return compare_natural(basename(a), basename(b)) < 0;
}

}  // namespace

namespace autoware::planning_debug_tools
{

std::vector<std::string> PerceptionReplayerCommon::find_rosbag_files(
  const std::string & directory_path, const std::string & rosbag_format) const
{
  const std::string extension = (rosbag_format == "mcap") ? ".mcap" : ".db3";
  std::vector<std::string> rosbag_files;

  rosbag2_storage::MetadataIo metadata_io;
  if (metadata_io.metadata_file_exists(directory_path)) {
    const auto metadata = metadata_io.read_metadata(directory_path);
    for (const auto & relative_path : metadata.relative_file_paths) {
      const std::filesystem::path full_path = std::filesystem::path(directory_path) / relative_path;
      if (std::filesystem::is_regular_file(full_path) && full_path.extension() == extension) {
        rosbag_files.push_back(full_path.string());
      }
    }
    if (!rosbag_files.empty()) {
      return rosbag_files;
    }
  }

  for (const auto & entry : std::filesystem::directory_iterator(directory_path)) {
    if (entry.is_regular_file() && entry.path().extension() == extension) {
      rosbag_files.push_back(entry.path().string());
    }
  }

  std::sort(rosbag_files.begin(), rosbag_files.end(), natural_path_less);

  return rosbag_files;
}

void PerceptionReplayerCommon::load_rosbag(
  const std::string & rosbag_path, const std::string & rosbag_format)
{
  std::cout << "Loading rosbag: " << rosbag_path << std::endl;

  auto reader = std::make_unique<rosbag2_cpp::Reader>();

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = rosbag_path;
  storage_options.storage_id = rosbag_format;

  reader->open(storage_options);

  // get topic metadata
  const auto topics = reader->get_all_topics_and_types();
  std::cout << "Found " << topics.size() << " topics in bag" << std::endl;

  // create type support map for deserialization
  std::unordered_map<std::string, std::shared_ptr<const rosidl_message_type_support_t>>
    type_support_map;
  std::unordered_map<std::string, std::shared_ptr<rcpputils::SharedLibrary>> type_support_libs;

  // try to load type support for each topic
  for (const auto & topic_meta : topics) {
    try {
      auto library = rclcpp::get_typesupport_library(topic_meta.type, "rosidl_typesupport_cpp");
      type_support_libs[topic_meta.name] = library;

#ifdef ROS_DISTRO_HUMBLE
      const rosidl_message_type_support_t * type_support =
        rclcpp::get_typesupport_handle(topic_meta.type, "rosidl_typesupport_cpp", *library);
#else
      const rosidl_message_type_support_t * type_support =
        rclcpp::get_message_typesupport_handle(topic_meta.type, "rosidl_typesupport_cpp", *library);
#endif

      if (type_support) {
        type_support_map[topic_meta.name] = std::shared_ptr<const rosidl_message_type_support_t>(
          type_support, [](const rosidl_message_type_support_t *) {});
      }
    } catch (const std::exception & e) {
      // skip topics with unknown message types
      std::cerr << "Warning: Could not load type support for topic " << topic_meta.name << " ("
                << topic_meta.type << ")" << std::endl;
    }
  }

  // topic_names
  const auto objects_topic = [&]() -> std::string {
    if (param_.tracked_object) {
      return "/perception/object_recognition/tracking/objects";
    } else {
      return "/perception/object_recognition/objects";
    }
  }();
  const std::string ego_odom_topic = "/localization/kinematic_state";
  const std::string traffic_signals_topic = "/perception/traffic_light_recognition/traffic_signals";
  const std::string occupancy_grid_topic = "/perception/occupancy_grid_map/map";
  const std::string route_topic = "/planning/mission_planning/route";
  const std::string route_state_topic = "/planning/mission_planning/state";

  // create topic filter
  rosbag2_storage::StorageFilter storage_filter;
  storage_filter.topics = {
    objects_topic,
    ego_odom_topic,
    traffic_signals_topic,
    occupancy_grid_topic,
  };

  if (param_.replay_route) {
    storage_filter.topics.push_back(route_topic);
    storage_filter.topics.push_back(route_state_topic);
  }

  // Add reference image topics to filter
  for (const auto & topic : param_.reference_image_topics) {
    storage_filter.topics.push_back(topic);
  }

  reader->set_filter(storage_filter);

  // read all messages
  while (reader->has_next()) {
    try {
      auto bag_message = reader->read_next();

      // deserialize the message if type support is available
      auto it = type_support_map.find(bag_message->topic_name);
      if (it != type_support_map.end() && it->second) {
        // deserialize ego_odom messages
        if (bag_message->topic_name == ego_odom_topic) {
          const auto ego_odom_msg =
            utils::deserialize_message<Odometry>(bag_message->serialized_data);
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          rosbag_ego_odom_data_.emplace_back(timestamp, *ego_odom_msg);
        }

        // deserialize objects messages
        if (bag_message->topic_name == objects_topic) {
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          if (param_.tracked_object) {
            const auto objects_msg =
              utils::deserialize_message<TrackedObjects>(bag_message->serialized_data);
            rosbag_tracked_objects_data_.emplace_back(timestamp, *objects_msg);
          } else {
            const auto objects_msg =
              utils::deserialize_message<PredictedObjects>(bag_message->serialized_data);
            rosbag_predicted_objects_data_.emplace_back(timestamp, *objects_msg);
          }
        }

        // deserialize traffic_signals messages
        if (bag_message->topic_name == traffic_signals_topic) {
          const auto traffic_signals_msg =
            utils::deserialize_message<TrafficLightGroupArray>(bag_message->serialized_data);
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          rosbag_traffic_signals_data_.emplace_back(timestamp, *traffic_signals_msg);
        }

        // deserialize occupancy_grid messages
        if (bag_message->topic_name == occupancy_grid_topic) {
          const auto occupancy_grid_msg =
            utils::deserialize_message<OccupancyGrid>(bag_message->serialized_data);
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          rosbag_occupancy_grid_data_.emplace_back(timestamp, *occupancy_grid_msg);
        }

        // deserialize route messages
        if (bag_message->topic_name == route_topic) {
          const auto route_msg =
            utils::deserialize_message<LaneletRoute>(bag_message->serialized_data);
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          rosbag_route_data_.emplace_back(timestamp, *route_msg);
        }

        // deserialize route_state messages
        if (bag_message->topic_name == route_state_topic) {
          const auto route_state_msg =
            utils::deserialize_message<RouteState>(bag_message->serialized_data);
          const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
          rosbag_route_state_data_.emplace_back(timestamp, *route_state_msg);
        }

        // deserialize reference image messages
        for (const auto & ref_topic : param_.reference_image_topics) {
          if (bag_message->topic_name == ref_topic) {
            const auto image_msg =
              utils::deserialize_message<CompressedImage>(bag_message->serialized_data);
            const rclcpp::Time timestamp(get_timestamp_ns(*bag_message));
            rosbag_reference_image_data_[ref_topic].emplace_back(timestamp, *image_msg);
            break;  // Found matching topic, no need to check others
          }
        }
      } else {
        // count messages that couldn't be deserialized
      }
    } catch (const std::exception & e) {
      std::cerr << "\nError reading message: " << e.what() << std::endl;
      continue;
    }
  }

  std::cout << "Finished loading rosbag: " << rosbag_path << std::endl;
}

PerceptionReplayerCommon::PerceptionReplayerCommon(
  const PerceptionReplayerCommonParam & param, const std::string & node_name,
  const rclcpp::NodeOptions & node_options)
: Node(node_name, node_options),
  param_(param),
  gen_(rd_()),
  uniform_dist_(0.0, 1.0),
  standard_dist_(0.0, 1.0)
{
  // check if rosbag_path is a directory or file
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
  std::cout << "Ended loading rosbag" << std::endl;

  // Assume kinematic ~50 Hz, perception 10 Hz: keep every 5th odom sample.
  constexpr size_t k_ego_odom_downsample_stride = 5;
  {
    const size_t before = rosbag_ego_odom_data_.size();
    std::vector<utils::DataStamped<Odometry>> downsampled;
    downsampled.reserve((before + k_ego_odom_downsample_stride - 1) / k_ego_odom_downsample_stride);
    for (size_t i = 0; i < before; i += k_ego_odom_downsample_stride) {
      downsampled.push_back(std::move(rosbag_ego_odom_data_[i]));
    }
    rosbag_ego_odom_data_ = std::move(downsampled);
    std::cout << "Downsampled ego_odom: " << before << " -> " << rosbag_ego_odom_data_.size()
              << " (stride=" << k_ego_odom_downsample_stride << ")" << std::endl;
  }

  // define topic names
  const std::string ego_odom_topic = "/localization/kinematic_state";
  const auto objects_topic = [&]() -> std::string {
    if (param_.tracked_object) {
      return "/perception/object_recognition/tracking/objects";
    } else {
      return "/perception/object_recognition/objects";
    }
  }();

  // Subscriber
  ego_odom_sub_ = this->create_subscription<Odometry>(
    ego_odom_topic, 1,
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

  if (param_.replay_route) {
    rclcpp::QoS transient_local_qos(1);
    transient_local_qos.transient_local();
    route_pub_ =
      this->create_publisher<LaneletRoute>("/planning/mission_planning/route", transient_local_qos);
    route_state_pub_ =
      this->create_publisher<RouteState>("/planning/mission_planning/state", transient_local_qos);
  }

  recorded_ego_as_initialpose_pub_ =
    this->create_publisher<PoseWithCovarianceStamped>("/initialpose", 1);
  goal_as_mission_planning_goal_pub_ =
    this->create_publisher<PoseStamped>("/planning/mission_planning/goal", 1);

  // Create reference image publishers (1:1 topic mapping)
  for (const auto & topic : param_.reference_image_topics) {
    reference_image_pubs_[topic] = this->create_publisher<CompressedImage>(topic, 1);
    RCLCPP_INFO(get_logger(), "Reference image enabled for topic: %s", topic.c_str());
  }

  // create timer to periodically check and kill online perception nodes (0.1 Hz)
  // Use Reentrant callback group to allow parallel execution with other timers
  callback_group_check_perception_ =
    this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  timer_check_perception_process_ = rclcpp::create_timer(
    this, get_clock(), std::chrono::seconds(10),
    std::bind(&PerceptionReplayerCommon::kill_online_perception_node, this),
    callback_group_check_perception_);
}

Odometry PerceptionReplayerCommon::find_ego_odom_by_timestamp(const rclcpp::Time & timestamp) const
{
  const size_t idx = utils::get_nearest_index(rosbag_ego_odom_data_, timestamp);
  return rosbag_ego_odom_data_.at(idx).second;
}

void PerceptionReplayerCommon::publish_topics_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp,
  const bool apply_noise)
{
  // for debugging
  recorded_ego_pub_->publish(find_ego_odom_by_timestamp(bag_timestamp));

  // publish objects
  const auto publish_objects = [&](auto & data) {
    const auto objects_msg = utils::find_message_by_timestamp(data, bag_timestamp);
    if (objects_msg.has_value()) {
      auto msg = objects_msg.value();
      if (apply_noise) {
        apply_perception_noise(msg);
      }
      msg.header.stamp = current_timestamp;
      using MessageType = std::decay_t<decltype(msg)>;
      if (
        auto publisher = std::dynamic_pointer_cast<rclcpp::Publisher<MessageType>>(objects_pub_)) {
        publisher->publish(msg);
      }
    }
  };

  if (param_.tracked_object) {
    publish_objects(rosbag_tracked_objects_data_);
  } else {
    publish_objects(rosbag_predicted_objects_data_);
  }

  publish_traffic_lights_at_timestamp(bag_timestamp, current_timestamp);

  // publish reference images
  publish_reference_images_at_timestamp(bag_timestamp, current_timestamp);

  // publish occupancy grid
  if (!rosbag_occupancy_grid_data_.empty()) {
    const size_t idx = utils::get_nearest_index(rosbag_occupancy_grid_data_, bag_timestamp);
    auto & msg = rosbag_occupancy_grid_data_[idx].second;
    msg.header.stamp = current_timestamp;
    occupancy_grid_pub_->publish(msg);
  }

  if (param_.replay_route) {
    publish_route_at_timestamp(bag_timestamp, current_timestamp);
  }
}

void PerceptionReplayerCommon::publish_traffic_lights_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  const auto traffic_signals_msg =
    utils::find_message_by_timestamp(rosbag_traffic_signals_data_, bag_timestamp);
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

void PerceptionReplayerCommon::publish_reference_images_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  for (const auto & topic : param_.reference_image_topics) {
    // Check if we have data for this topic
    auto it = rosbag_reference_image_data_.find(topic);
    if (it == rosbag_reference_image_data_.end() || it->second.empty()) {
      continue;
    }

    // Find nearest image by timestamp
    const auto image_msg = utils::find_message_by_timestamp(it->second, bag_timestamp);

    if (image_msg.has_value()) {
      auto msg = image_msg.value();
      msg.header.stamp = current_timestamp;
      reference_image_pubs_[topic]->publish(msg);
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "No reference image found for topic %s at timestamp %f",
        topic.c_str(), bag_timestamp.seconds());
    }
  }
}

void PerceptionReplayerCommon::publish_route_at_timestamp(
  const rclcpp::Time & bag_timestamp, const rclcpp::Time & current_timestamp)
{
  // Helper: find the index of the last message at or before bag_timestamp.
  // Returns nullopt if there is no such message.
  auto find_last_before = [](const auto & data, const rclcpp::Time & ts) -> std::optional<size_t> {
    if (data.empty() || data.front().first > ts) {
      return std::nullopt;
    }
    // binary search for rightmost element with timestamp <= ts
    size_t lo = 0;
    size_t hi = data.size();
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      if (data[mid].first <= ts) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo - 1;
  };

  // route
  if (!rosbag_route_data_.empty()) {
    const auto idx = find_last_before(rosbag_route_data_, bag_timestamp);
    if (idx.has_value() && last_published_route_idx_ != idx.value()) {
      auto route_msg = rosbag_route_data_[idx.value()].second;
      route_msg.header.stamp = current_timestamp;
      route_pub_->publish(route_msg);
      last_published_route_idx_ = idx.value();
    }
  }

  // route state: publish when bag index changes, or re-publish every 5s (replay time)
  if (!rosbag_route_state_data_.empty()) {
    const auto idx = find_last_before(rosbag_route_state_data_, bag_timestamp);
    if (idx.has_value()) {
      const bool index_changed = !last_published_route_state_idx_.has_value() ||
                                 last_published_route_state_idx_.value() != idx.value();
      constexpr double k_route_state_republish_period_sec = 10.0;
      const bool period_elapsed =
        !last_route_state_publish_replay_time_.has_value() ||
        (current_timestamp - last_route_state_publish_replay_time_.value()).seconds() >=
          k_route_state_republish_period_sec;

      if (index_changed || period_elapsed) {
        auto route_state_msg = rosbag_route_state_data_[idx.value()].second;
        route_state_msg.stamp = current_timestamp;
        route_state_pub_->publish(route_state_msg);
        last_published_route_state_idx_ = idx.value();
        last_route_state_publish_replay_time_ = current_timestamp;
      }
    }
  }
}

void PerceptionReplayerCommon::publish_recorded_ego_pose(rclcpp::Time bag_timestamp)
{
  const auto ego_odom = find_ego_odom_by_timestamp(bag_timestamp);
  const auto initialpose =
    utils::make_map_initial_pose(ego_odom.pose.pose, this->get_clock()->now());
  recorded_ego_as_initialpose_pub_->publish(initialpose);

  RCLCPP_INFO(get_logger(), "Published recorded ego pose as /initialpose");
}

void PerceptionReplayerCommon::publish_goal_pose()
{
  const auto ego_pose = find_ego_odom_by_timestamp(get_bag_end_timestamp());

  PoseStamped goal_pose;
  goal_pose.header.stamp = this->get_clock()->now();
  goal_pose.header.frame_id = "map";
  goal_pose.pose = ego_pose.pose.pose;

  goal_as_mission_planning_goal_pub_->publish(goal_pose);
  RCLCPP_INFO(get_logger(), "Published last recorded ego pose as /planning/mission_planning/goal");
}

void PerceptionReplayerCommon::publish_localization_and_route()
{
  publish_recorded_ego_pose(get_bag_start_time());
  // temporarily add a sleep because sometimes the route is not generated correctly without it.
  // Need to consider a proper solution.
  rclcpp::sleep_for(std::chrono::seconds(2));
  publish_goal_pose();
}

void PerceptionReplayerCommon::on_ego_odom(const Odometry::SharedPtr msg)
{
  ego_odom_ = msg;
}

void PerceptionReplayerCommon::kill_online_perception_node()
{
  // kill the object recognition node
  if (param_.tracked_object && !rosbag_tracked_objects_data_.empty()) {
    kill_process("multi_object_tracker");
  } else if (!rosbag_predicted_objects_data_.empty()) {
    kill_process("map_based_prediction");
  }

  // unload the occupancy grid map node only if rosbag contains occupancy grid data
  if (!rosbag_occupancy_grid_data_.empty()) {
    unload_component("/pointcloud_container", "occupancy_grid_map_node");
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

}  // namespace autoware::planning_debug_tools
