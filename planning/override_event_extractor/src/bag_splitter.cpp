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
  const std::string & output_dir)
{
  std::vector<std::string> output_files;
  const auto base_name = getBagBaseName(input_bag);

  for (const auto & event : events) {
    const auto output_filename = base_name + "_or_" + std::to_string(event.index) + ".mcap";
    const auto output_path = std::filesystem::path(output_dir) / output_filename;

    extractSegment(input_bag, event, output_path.string());
    output_files.push_back(output_filename);
  }

  return output_files;
}

void BagSplitter::extractSegment(
  const std::string & input_bag, const OverrideEvent & event, const std::string & output_path)
{
  rosbag2_cpp::Reader reader;
  reader.open(input_bag);

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = output_path;
  storage_options.storage_id = config_.storage_id;

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.output_serialization_format = config_.serialization_format;

  rosbag2_cpp::Writer writer;
  writer.open(storage_options, converter_options);

  std::unordered_set<std::string> created_topics;

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
          topic_metadata.serialization_format = config_.serialization_format;

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
