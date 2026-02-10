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

#include "bag_splitter.hpp"

#include <rcutils/allocator.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>

namespace override_event_extractor
{

BagSplitter::BagSplitter(const SplitterConfig & config) : config_(config)
{
  preserved_topics_set_ =
    std::unordered_set<std::string>(config_.preserved_topics.begin(), config_.preserved_topics.end());
}

std::vector<std::string> BagSplitter::split(
  const std::string & input_bag, const std::vector<OverrideEvent> & events,
  const std::string & output_dir, const RouteMessage & route_msg)
{
  std::vector<std::string> output_files;
  const auto base_name = getBagBaseName(input_bag);

  for (const auto & event : events) {
    const auto output_filename = base_name + "_or_" + std::to_string(event.index) + ".mcap";
    const auto output_path = std::filesystem::path(output_dir) / output_filename;

    extractSegment(input_bag, event, output_path.string(), route_msg);
    output_files.push_back(output_filename);
  }

  return output_files;
}

void BagSplitter::extractSegment(
  const std::string & input_bag, const OverrideEvent & event, const std::string & output_path,
  const RouteMessage & route_msg)
{
  rosbag2_cpp::Reader reader;
  reader.open(input_bag);

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = output_path;
  storage_options.storage_id = config_.storage_id;

  rosbag2_cpp::Writer writer;
  writer.open(storage_options);

  std::unordered_set<std::string> created_topics;

  // Inject route message first if available
  if (route_msg.valid && route_msg.message) {
    // Create topic for route
    const auto & topics = reader.get_all_topics_and_types();
    for (const auto & topic_meta : topics) {
      if (topic_meta.name == config_.route_topic) {
        rosbag2_storage::TopicMetadata topic_metadata;
        topic_metadata.name = topic_meta.name;
        topic_metadata.type = topic_meta.type;
        topic_metadata.serialization_format = topic_meta.serialization_format;

        writer.create_topic(topic_metadata);
        created_topics.insert(topic_meta.name);

        // Create a deep copy of the route message with new timestamp
        auto route_copy = std::make_shared<rosbag2_storage::SerializedBagMessage>();
        route_copy->topic_name = route_msg.message->topic_name;
        route_copy->time_stamp = event.extended_range.start_ns;
        route_copy->serialized_data = std::make_shared<rcutils_uint8_array_t>();
        *route_copy->serialized_data = rcutils_get_zero_initialized_uint8_array();

        // Allocate and copy data
        auto allocator = rcutils_get_default_allocator();
        auto ret = rcutils_uint8_array_init(
          route_copy->serialized_data.get(), route_msg.message->serialized_data->buffer_length,
          &allocator);
        if (ret == RCUTILS_RET_OK) {
          memcpy(
            route_copy->serialized_data->buffer, route_msg.message->serialized_data->buffer,
            route_msg.message->serialized_data->buffer_length);
          route_copy->serialized_data->buffer_length =
            route_msg.message->serialized_data->buffer_length;

          writer.write(route_copy);
        }
        break;
      }
    }
  }

  while (reader.has_next()) {
    auto bag_message = reader.read_next();

    if (!inRange(bag_message->time_stamp, event.extended_range)) {
      continue;
    }

    if (!shouldPreserveTopic(bag_message->topic_name)) {
      continue;
    }

    if (created_topics.find(bag_message->topic_name) == created_topics.end()) {
      const auto & topics = reader.get_all_topics_and_types();
      for (const auto & topic_meta : topics) {
        if (topic_meta.name == bag_message->topic_name) {
          rosbag2_storage::TopicMetadata topic_metadata;
          topic_metadata.name = topic_meta.name;
          topic_metadata.type = topic_meta.type;
          topic_metadata.serialization_format = topic_meta.serialization_format;

          writer.create_topic(topic_metadata);
          created_topics.insert(bag_message->topic_name);
          break;
        }
      }
    }

    writer.write(bag_message);
  }
}

bool BagSplitter::inRange(int64_t timestamp, const TimeRange & range) const
{
  return timestamp >= range.start_ns && timestamp <= range.end_ns;
}

bool BagSplitter::shouldPreserveTopic(const std::string & topic_name) const
{
  return preserved_topics_set_.find(topic_name) != preserved_topics_set_.end();
}

std::string BagSplitter::getBagBaseName(const std::string & bag_path) const
{
  std::filesystem::path p(bag_path);
  auto stem = p.stem().string();

  if (stem.size() >= 2 && stem.substr(stem.size() - 2) == "_0") {
    stem = stem.substr(0, stem.length() - 2);
  }

  return stem;
}

}  // namespace override_event_extractor
