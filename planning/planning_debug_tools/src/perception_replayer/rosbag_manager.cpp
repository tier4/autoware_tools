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

#include "rosbag_manager.hpp"

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rcpputils/shared_library.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace autoware::planning_debug_tools
{

RosbagManager::RosbagManager(rclcpp::Logger logger, bool tracked_object, const RosbagManagerParam & param)
: logger_(logger), tracked_object_(tracked_object), param_(param)
{
  // Initialize topic names
  ego_odom_topic_ = "/localization/kinematic_state";
  objects_topic_ = tracked_object ? "/perception/object_recognition/tracking/objects"
                                  : "/perception/object_recognition/objects";
  traffic_signals_topic_ = "/perception/traffic_light_recognition/traffic_signals";
  occupancy_grid_topic_ = "/perception/occupancy_grid_map/map";
  pointcloud_topic_ = "/perception/obstacle_segmentation/pointcloud";
  route_state_topic_ = "/planning/mission_planning/state";
  route_topic_ = "/planning/mission_planning/route";
}

void RosbagManager::load_type_support(const rosbag2_storage::TopicMetadata & topic_meta)
{
  if (type_support_map_.find(topic_meta.name) == type_support_map_.end()) {
    try {
      auto library =
        rosbag2_cpp::get_typesupport_library(topic_meta.type, "rosidl_typesupport_cpp");
      type_support_libs_[topic_meta.name] = library;

      const rosidl_message_type_support_t * type_support =
        rosbag2_cpp::get_typesupport_handle(topic_meta.type, "rosidl_typesupport_cpp", library);

      if (type_support) {
        type_support_map_[topic_meta.name] = std::shared_ptr<const rosidl_message_type_support_t>(
          type_support, [](const rosidl_message_type_support_t *) {});
      }
    } catch (const std::exception & e) {
      // skip topics with unknown message types
    }
  }
}

void RosbagManager::process_message(
  const rclcpp::Time & msg_timestamp, const std::string & topic_name,
  const std::shared_ptr<rcutils_uint8_array_t> & serialized_data)
{
  auto it = type_support_map_.find(topic_name);
  if (it != type_support_map_.end() && it->second) {
    if (topic_name == ego_odom_topic_) {
      const auto ego_odom_msg = utils::deserialize_message<Odometry>(serialized_data);
      rosbag_ego_odom_data_.emplace_back(msg_timestamp, *ego_odom_msg);
    } else if (topic_name == objects_topic_) {
      if (tracked_object_) {
        const auto objects_msg = utils::deserialize_message<TrackedObjects>(serialized_data);
        rosbag_tracked_objects_data_.emplace_back(msg_timestamp, *objects_msg);
      } else {
        const auto objects_msg = utils::deserialize_message<PredictedObjects>(serialized_data);
        rosbag_predicted_objects_data_.emplace_back(msg_timestamp, *objects_msg);
      }
    } else if (topic_name == traffic_signals_topic_) {
      const auto traffic_signals_msg =
        utils::deserialize_message<TrafficLightGroupArray>(serialized_data);
      rosbag_traffic_signals_data_.emplace_back(msg_timestamp, *traffic_signals_msg);
    } else if (topic_name == occupancy_grid_topic_) {
      const auto occupancy_grid_msg = utils::deserialize_message<OccupancyGrid>(serialized_data);
      rosbag_occupancy_grid_data_.emplace_back(msg_timestamp, *occupancy_grid_msg);
    } else if (topic_name == pointcloud_topic_) {
      const auto pointcloud_msg = utils::deserialize_message<PointCloud2>(serialized_data);
      rosbag_pointcloud_data_.emplace_back(msg_timestamp, *pointcloud_msg);
    } else if (topic_name == route_state_topic_) {
      const auto route_state_msg = utils::deserialize_message<RouteState>(serialized_data);
      rosbag_route_state_data_.emplace_back(msg_timestamp, *route_state_msg);
    } else if (topic_name == route_topic_) {
      const auto route_msg = utils::deserialize_message<LaneletRoute>(serialized_data);
      rosbag_route_data_.emplace_back(msg_timestamp, *route_msg);
    }
  }
}

void RosbagManager::load_rosbag(const std::string & rosbag_path)
{
  std::cout << "Load rosbag: " << rosbag_path << std::endl;

  auto reader = std::make_unique<rosbag2_cpp::Reader>();

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = rosbag_path;
  storage_options.storage_id = param_.rosbag_format;

  reader->open(storage_options);

  // load type support for each topic
  const auto topics = reader->get_all_topics_and_types();
  for (const auto & topic_meta : topics) {
    load_type_support(topic_meta);
  }

  // create topic filter
  rosbag2_storage::StorageFilter storage_filter;
  storage_filter.topics = {
    objects_topic_,
    ego_odom_topic_,
    traffic_signals_topic_,
    occupancy_grid_topic_,
    pointcloud_topic_,
    route_state_topic_,
    route_topic_,
  };
  reader->set_filter(storage_filter);

  // read all messages
  while (reader->has_next()) {
    try {
      auto bag_message = reader->read_next();
      const rclcpp::Time timestamp(bag_message->time_stamp);
      process_message(timestamp, bag_message->topic_name, bag_message->serialized_data);
    } catch (const std::exception & e) {
      std::cerr << "\nError reading message: " << e.what() << std::endl;
      continue;
    }
  }
}

void RosbagManager::initialize(const std::string & rosbag_path)
{
  // Collect rosbag files
  std::vector<std::string> rosbag_files;
  if (std::filesystem::is_directory(rosbag_path)) {
    // Find all rosbag files in directory matching the format
    const std::string extension = (param_.rosbag_format == "mcap") ? ".mcap" : ".db3";
    for (const auto & entry : std::filesystem::directory_iterator(rosbag_path)) {
      if (entry.is_regular_file() && entry.path().extension() == extension) {
        rosbag_files.push_back(entry.path().string());
      }
    }
    std::sort(rosbag_files.begin(), rosbag_files.end());
  } else {
    rosbag_files.push_back(rosbag_path);
  }

  // Use cache-based loading or load all data at once
  if (cache_config_.enabled) {
    RCLCPP_INFO(logger_, "Use cache-based loading mode to load %zu bag files", rosbag_files.size());
    initialize_rosbag_cache(rosbag_files);
  } else {
    RCLCPP_INFO(logger_, "Load all data at once from %zu bag files", rosbag_files.size());
    for (size_t i = 0; i < rosbag_files.size(); ++i) {
      RCLCPP_INFO(logger_, "Load bag file %zu/%zu: %s", i + 1, rosbag_files.size(), rosbag_files[i].c_str());
      load_rosbag(rosbag_files[i]);
    }
  }
}

Odometry RosbagManager::find_ego_odom_by_timestamp(const rclcpp::Time & timestamp) const
{
  const size_t idx = utils::get_nearest_index(rosbag_ego_odom_data_, timestamp);
  return rosbag_ego_odom_data_.at(idx).second;
}

void RosbagManager::initialize_rosbag_cache(const std::vector<std::string> & rosbag_files) // TODO 从这往下看。odashima的pr merge后用这个rosbag_manager重构代码
{
  // Clear existing readers
  rosbag_readers_.clear();

  if (rosbag_files.empty()) {
    return;
  }

  RCLCPP_INFO(logger_, "Initializing cache for %zu rosbag files", rosbag_files.size());

  // Clear cached rosbag infos
  cached_rosbag_infos_.clear();

  // Initialize readers for all rosbag files and record time ranges
  for (const auto & file_path : rosbag_files) {
    auto reader = std::make_unique<rosbag2_cpp::Reader>();

      rosbag2_storage::StorageOptions storage_options;
      storage_options.uri = file_path;
      storage_options.storage_id = info.rosbag_format;

    try {
      reader->open(storage_options);

      // Get topic metadata and load type support
      const auto topics = reader->get_all_topics_and_types();

      for (const auto & topic_meta : topics) {
        load_type_support(topic_meta);
      }

      // Set topic filter
      rosbag2_storage::StorageFilter storage_filter;
      storage_filter.topics = {
        ego_odom_topic_,
        objects_topic_,
        traffic_signals_topic_,
        occupancy_grid_topic_,
        pointcloud_topic_,
        route_state_topic_,
        route_topic_,
      };
      reader->set_filter(storage_filter);

      // Read first and last messages to get time range
      rclcpp::Time start_time = rclcpp::Time(0);
      rclcpp::Time end_time = rclcpp::Time(0);

      if (reader->has_next()) {
        // Get first message timestamp
        auto first_message = reader->read_next();
        start_time = rclcpp::Time(first_message->time_stamp);
        end_time = start_time;  // Initialize with first message

        // Read through all messages to find the last one
        while (reader->has_next()) {
          auto message = reader->read_next();
          end_time = rclcpp::Time(message->time_stamp);
        }

        // Reset reader to beginning for actual use
        reader.reset();
        reader = std::make_unique<rosbag2_cpp::Reader>();
        reader->open(storage_options);
        reader->set_filter(storage_filter);
      } else {
        // Empty rosbag, use zero timestamps
        start_time = rclcpp::Time(0);
        end_time = rclcpp::Time(0);
      }

      // Store rosbag info with time range
      RosbagInfo info;
      info.file_path = file_path;
      info.start_time = start_time;
      info.end_time = end_time;
      cached_rosbag_infos_.push_back(info);

      rosbag_readers_.push_back(std::move(reader));
      RCLCPP_INFO(
        logger_, "Initialized reader for: %s [%.3f - %.3f]", file_path.c_str(),
        start_time.seconds(), end_time.seconds());
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "Failed to open rosbag file %s: %s", file_path.c_str(), e.what());
    }
  }

  last_loaded_timestamp_ = rclcpp::Time(0);
  rosbag_cache_initialized_ = !rosbag_readers_.empty();


  RCLCPP_INFO(logger_, "Rosbag cache initialized with %zu readers", rosbag_readers_.size());
  
  // Load all ego_odom and route_state data immediately (frequently used, so load all at once)
  load_all_ego_odom_data(rosbag_files);
  load_all_route_state_data(rosbag_files);
}

void RosbagManager::load_cache_data(const rclcpp::Time & base_timestamp)
{
  if (!rosbag_cache_initialized_ || rosbag_readers_.empty()) {
    return;
  }

  if (base_timestamp.nanoseconds() == 0) {
    return;  // No current timestamp yet
  }

  // Update internal cache timestamp
  current_cache_timestamp_ = base_timestamp;

  const rclcpp::Time target_end_time =
    base_timestamp + rclcpp::Duration::from_seconds(param_.cache_window_after_sec);
  const rclcpp::Time cleanup_before_time =
    base_timestamp - rclcpp::Duration::from_seconds(param_.cache_window_before_sec);

  // Check if current timestamp is outside the cache window
  // Find the earliest and latest timestamps in cache
  rclcpp::Time cache_earliest = rclcpp::Time::max();
  rclcpp::Time cache_latest = rclcpp::Time(0);

  auto find_time_range = [&](const auto & data_vec) {
    if (!data_vec.empty()) {
      cache_earliest = std::min(cache_earliest, data_vec.front().first);
      cache_latest = std::max(cache_latest, data_vec.back().first);
    }
  };

  // Note: rosbag_ego_odom_data_ and rosbag_route_state_data_ are not included in cache window check
  find_time_range(rosbag_tracked_objects_data_);
  find_time_range(rosbag_predicted_objects_data_);
  find_time_range(rosbag_traffic_signals_data_);
  find_time_range(rosbag_occupancy_grid_data_);
  find_time_range(rosbag_pointcloud_data_);
  find_time_range(rosbag_route_data_);

  // Check if current timestamp is outside the cache window
  // Only backward time jump requires reset (rosbag2 doesn't support backward seek)
  const bool cache_empty = cache_earliest == rclcpp::Time::max();
  const bool time_jumped_backward = !cache_empty && base_timestamp < cleanup_before_time;

  if (time_jumped_backward) {
    RCLCPP_INFO(
      logger_,
      "Backward time jump detected (current: %.3f, cache range: [%.3f, %.3f]). Resetting cache...",
      base_timestamp.seconds(), cache_empty ? 0.0 : cache_earliest.seconds(),
      cache_empty ? 0.0 : cache_latest.seconds());

    // Clear all cached data (except ego_odom and route_state which are always fully loaded)
    rosbag_tracked_objects_data_.clear();
    rosbag_predicted_objects_data_.clear();
    rosbag_traffic_signals_data_.clear();
    rosbag_occupancy_grid_data_.clear();
    rosbag_pointcloud_data_.clear();
    rosbag_route_data_.clear();

    // Find which rosbag files contain the target timestamp
    std::vector<std::string> rosbag_files_to_load;
    for (const auto & info : cached_rosbag_infos_) {
      if (base_timestamp >= info.start_time && base_timestamp <= info.end_time) {
        rosbag_files_to_load.push_back(info.file_path);
      }
    }

    // If no specific rosbag found, reload all (fallback)
    if (rosbag_files_to_load.empty()) {
      for (const auto & info : cached_rosbag_infos_) {
        rosbag_files_to_load.push_back(info.file_path);
      }
    }

    // Close and clear existing readers
    rosbag_readers_.clear();

    // Reinitialize readers from the beginning (or from relevant rosbags)
    if (!rosbag_files_to_load.empty()) {
      // Clear ego_odom_data and route_state_data before reloading (they will be fully reloaded)
      rosbag_ego_odom_data_.clear();
      rosbag_route_state_data_.clear();
      initialize_rosbag_cache(rosbag_files_to_load);
    }
  }

  // Load data from all rosbag readers sequentially
  size_t loaded_count = 0;

  for (auto & reader : rosbag_readers_) {
    while (reader->has_next() && last_loaded_timestamp_ < target_end_time) {
      try {
        auto bag_message = reader->read_next();
        const rclcpp::Time msg_timestamp(bag_message->time_stamp);

        // Skip messages outside the cache window
        if (msg_timestamp < cleanup_before_time) {
          // Message is too old, skip it (but continue reading)
          continue;
        }
        if (msg_timestamp > target_end_time) {
          // Message is beyond cache window, stop reading from this reader
          break;
        }

        // Message is within cache window, process it
        // Skip ego_odom and route_state messages as they are already fully loaded
        if (bag_message->topic_name != ego_odom_topic_ && 
            bag_message->topic_name != route_state_topic_) {
          process_message(msg_timestamp, bag_message->topic_name, bag_message->serialized_data);
          loaded_count++;
        }
        last_loaded_timestamp_ = std::max(last_loaded_timestamp_, msg_timestamp);
      } catch (const std::exception & e) {
        break;
      }
    }
  }

  // Clean up data older than cleanup_before_time
  // Note: rosbag_ego_odom_data_ and rosbag_route_state_data_ are not cleaned up as they are always fully loaded
  auto remove_before = [&cleanup_before_time](auto & data_vec) {
    data_vec.erase(
      std::remove_if(
        data_vec.begin(), data_vec.end(),
        [&cleanup_before_time](const auto & item) { return item.first < cleanup_before_time; }),
      data_vec.end());
  };

  remove_before(rosbag_tracked_objects_data_);
  remove_before(rosbag_predicted_objects_data_);
  remove_before(rosbag_traffic_signals_data_);
  remove_before(rosbag_occupancy_grid_data_);
  remove_before(rosbag_pointcloud_data_);
  remove_before(rosbag_route_data_);

  if (loaded_count > 0) {
    RCLCPP_DEBUG(
      logger_, "Loaded %zu messages, cache window: [%.3f, %.3f]", loaded_count,
      cleanup_before_time.seconds(), target_end_time.seconds());
  }
}

rclcpp::Time RosbagManager::get_bag_start_time() const
{
  if (rosbag_ego_odom_data_.empty()) {
    throw std::runtime_error("No ego odom data available");
  }
  return rosbag_ego_odom_data_.front().first;
}

rclcpp::Time RosbagManager::get_bag_end_timestamp() const
{
  if (rosbag_ego_odom_data_.empty()) {
    throw std::runtime_error("No ego odom data available");
  }
  return rosbag_ego_odom_data_.back().first;
}

void RosbagManager::load_all_ego_odom_data(const std::vector<std::string> & rosbag_files)
{
  RCLCPP_INFO(logger_, "Loading all ego_odom data from %zu rosbag files", rosbag_files.size());
  
  for (const auto & file_path : rosbag_files) {
    auto reader = std::make_unique<rosbag2_cpp::Reader>();
    
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = file_path;
    storage_options.storage_id = param_.rosbag_format;
    
    try {
      reader->open(storage_options);
      
      // Set topic filter to only ego_odom topic
      rosbag2_storage::StorageFilter storage_filter;
      storage_filter.topics = {ego_odom_topic_};
      reader->set_filter(storage_filter);
      
      // Read all ego_odom messages
      while (reader->has_next()) {
        try {
          auto bag_message = reader->read_next();
          const rclcpp::Time timestamp(bag_message->time_stamp);
          process_message(timestamp, bag_message->topic_name, bag_message->serialized_data);
        } catch (const std::exception & e) {
          continue;
        }
      }
      
      RCLCPP_DEBUG(logger_, "Loaded ego_odom data from: %s", file_path.c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "Failed to load ego_odom from %s: %s", file_path.c_str(), e.what());
    }
  }
  
  RCLCPP_INFO(logger_, "Loaded %zu ego_odom messages in total", rosbag_ego_odom_data_.size());
}

void RosbagManager::load_all_route_state_data(const std::vector<std::string> & rosbag_files)
{
  RCLCPP_INFO(logger_, "Loading all route_state data from %zu rosbag files", rosbag_files.size());
  
  for (const auto & file_path : rosbag_files) {
    auto reader = std::make_unique<rosbag2_cpp::Reader>();
    
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = file_path;
    storage_options.storage_id = param_.rosbag_format;
    
    try {
      reader->open(storage_options);
      
      // Set topic filter to only route_state topic
      rosbag2_storage::StorageFilter storage_filter;
      storage_filter.topics = {route_state_topic_};
      reader->set_filter(storage_filter);
      
      // Read all route_state messages
      while (reader->has_next()) {
        try {
          auto bag_message = reader->read_next();
          const rclcpp::Time timestamp(bag_message->time_stamp);
          process_message(timestamp, bag_message->topic_name, bag_message->serialized_data);
        } catch (const std::exception & e) {
          continue;
        }
      }
      
      RCLCPP_DEBUG(logger_, "Loaded route_state data from: %s", file_path.c_str());
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "Failed to load route_state from %s: %s", file_path.c_str(), e.what());
    }
  }
  
  RCLCPP_INFO(logger_, "Loaded %zu route_state messages in total", rosbag_route_state_data_.size());
}

}  // namespace autoware::planning_debug_tools

