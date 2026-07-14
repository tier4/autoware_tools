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

#include "perception_reproducer.hpp"

#include "utils.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace autoware::planning_debug_tools
{
namespace
{
constexpr double k_ego_stopped_speed_eps = 0.1;  // [m/s]
constexpr double k_perturb_backoff_s = 5.0;
constexpr double k_perception_hz = 10.0;
// Reject bag odom whose heading is opposite to the reference pose (> 90 deg).
constexpr double k_max_heading_diff_for_nearby_rad = M_PI / 2.0;
}  // namespace

PerceptionReproducer::PerceptionReproducer(
  const PerceptionReproducerParam & param, const rclcpp::NodeOptions & node_options)
: PerceptionReplayerCommon(param, "perception_reproducer", node_options), param_(param)
{
  RCLCPP_INFO(get_logger(), "Starting PerceptionReproducer initialization");

  declare_parameter<bool>("auto_tackle_stuck", param_.auto_tackle_stuck);
  declare_parameter<double>("stuck_duration_s", param_.stuck_duration_s);
  declare_parameter<double>("expand_radius_scale", param_.expand_radius_scale);
  declare_parameter<double>("perturb_distance_m", param_.perturb_distance_m);
  declare_parameter<bool>("output_metrics", param_.output_metrics);

  // subscription for /initialpose3d to refresh cool down and sync bag anchor
  sub_init_pos_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/initialpose3d", 1,
    std::bind(&PerceptionReproducer::on_pose_reset, this, std::placeholders::_1));

  sub_autoware_state_ = this->create_subscription<autoware_system_msgs::msg::AutowareState>(
    "/autoware/state", 10,
    std::bind(&PerceptionReproducer::on_autoware_state, this, std::placeholders::_1));

  initialize_client_ = this->create_client<autoware_localization_msgs::srv::InitializeLocalization>(
    "/localization/initialize");

  metrics_pub_ = this->create_publisher<tier4_metric_msgs::msg::MetricArray>("~/metrics", 1);

  // Fixed 10 Hz loop; ego_odom is already downsampled to perception rate at load.
  // Use MutuallyExclusive group so MultiThreadedExecutor cannot overlap on_timer.
  constexpr double timer_period_s = 1.0 / k_perception_hz;
  RCLCPP_INFO(get_logger(), "Creating main timer at %.0f Hz (fixed)", k_perception_hz);
  timer_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  timer_ = rclcpp::create_timer(
    this, get_clock(), std::chrono::duration<double>(timer_period_s),
    std::bind(&PerceptionReproducer::on_timer, this), timer_callback_group_);

  if (param_.publish_route) {
    publish_recorded_ego_pose(get_bag_start_time());
    // temporarily add a sleep because sometimes the route is not generated correctly without it.
    // Need to consider a proper solution.
    rclcpp::sleep_for(std::chrono::seconds(2));
    publish_goal_pose();
  }

  RCLCPP_INFO(
    get_logger(),
    "auto_tackle_stuck=%s stuck_duration_s=%.3f expand_radius_scale=%.3f "
    "perturb_distance_m=%.3f output_metrics=%s",
    get_parameter("auto_tackle_stuck").as_bool() ? "true" : "false",
    get_parameter("stuck_duration_s").as_double(), get_parameter("expand_radius_scale").as_double(),
    get_parameter("perturb_distance_m").as_double(),
    get_parameter("output_metrics").as_bool() ? "true" : "false");
  RCLCPP_INFO(get_logger(), "PerceptionReproducer initialization completed");
}

PerceptionReproducer::~PerceptionReproducer()
{
  flush_state_duration(this->get_clock()->now());
  dump_metrics_json();
}

void PerceptionReproducer::reset_reproduce_tracking(
  const std::optional<rclcpp::Time> & new_bag_timestamp)
{
  cool_down_indices_.clear();
  last_sequenced_ego_pose_.reset();
  reproduce_sequence_indices_.clear();
  if (new_bag_timestamp.has_value()) {
    last_published_timestamp_ = new_bag_timestamp;
  }
}

void PerceptionReproducer::rebuild_reproduce_sequence(
  const geometry_msgs::msg::Pose & ego_pose, const double search_radius,
  const rclcpp::Time & current_timestamp)
{
  last_sequenced_ego_pose_ = ego_pose;

  const auto nearest_ego_odom_idx = find_nearest_ego_odom_index(ego_pose);
  std::vector<geometry_msgs::msg::Pose> nearby_ego_odom_poses;

  const auto nearby_indices_initial = find_nearby_ego_odom_indices({ego_pose}, search_radius);
  if (!nearby_indices_initial.empty()) {
    for (const auto idx : nearby_indices_initial) {
      nearby_ego_odom_poses.push_back(rosbag_ego_odom_data_[idx].second.pose.pose);
    }
  } else {
    nearby_ego_odom_poses.push_back(rosbag_ego_odom_data_[nearest_ego_odom_idx].second.pose.pose);
  }

  auto ego_odom_indices = find_nearby_ego_odom_indices(nearby_ego_odom_poses, search_radius);

  while (!cool_down_indices_.empty()) {
    const auto last_timestamp = ego_odom_id2last_published_timestamp_[cool_down_indices_.front()];
    if ((current_timestamp - last_timestamp).seconds() > param_.reproduce_cool_down) {
      cool_down_indices_.pop_front();
    } else {
      break;
    }
  }

  ego_odom_indices.erase(
    std::remove_if(
      ego_odom_indices.begin(), ego_odom_indices.end(),
      [this](const size_t idx) {
        return std::find(cool_down_indices_.begin(), cool_down_indices_.end(), idx) !=
               cool_down_indices_.end();
      }),
    ego_odom_indices.end());

  std::sort(ego_odom_indices.begin(), ego_odom_indices.end());
  reproduce_sequence_indices_ =
    std::deque<size_t>(ego_odom_indices.begin(), ego_odom_indices.end());
}

void PerceptionReproducer::on_autoware_state(
  const autoware_system_msgs::msg::AutowareState::SharedPtr msg)
{
  autoware_state_.store(msg->state);
}

void PerceptionReproducer::on_pose_reset(
  const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  std::optional<rclcpp::Time> new_bag_timestamp;
  if (!last_published_timestamp_.has_value()) {
    const auto nearest_ego_odom_idx = find_nearest_ego_odom_index(msg->pose.pose);
    new_bag_timestamp = rosbag_ego_odom_data_[nearest_ego_odom_idx].first;
  }
  reset_reproduce_tracking(new_bag_timestamp);
  tried_expand_while_stopped_ = false;
  stuck_since_.reset();

  RCLCPP_INFO(get_logger(), "Cool down indices and last sequenced pose cleared by /initialpose3d");
}

std::optional<size_t> PerceptionReproducer::find_perturb_index_along_bag(
  const size_t start_idx, const double distance_m) const
{
  if (rosbag_ego_odom_data_.empty() || start_idx >= rosbag_ego_odom_data_.size()) {
    return std::nullopt;
  }
  if (distance_m <= 0.0 || start_idx + 1 >= rosbag_ego_odom_data_.size()) {
    return start_idx;
  }

  double accumulated = 0.0;
  for (size_t i = start_idx; i + 1 < rosbag_ego_odom_data_.size(); ++i) {
    const auto & p0 = rosbag_ego_odom_data_[i].second.pose.pose.position;
    const auto & p1 = rosbag_ego_odom_data_[i + 1].second.pose.pose.position;
    accumulated += utils::calculate_distance_2d(p0, p1);
    if (accumulated >= distance_m) {
      return i + 1;
    }
  }

  RCLCPP_WARN(
    get_logger(),
    "Could not accumulate %.3f m along bag pose list from idx=%zu (reached end, used last pose)",
    distance_m, start_idx);
  return rosbag_ego_odom_data_.size() - 1;
}

bool PerceptionReproducer::speed_gap_forces_repeat(
  const nav_msgs::msg::Odometry & ego_odom, const geometry_msgs::msg::Pose & ego_pose) const
{
  if (reproduce_sequence_indices_.empty()) {
    return true;
  }

  const double ego_speed = utils::calculate_speed_2d(ego_odom.twist.twist);
  const auto & bag_odom = rosbag_ego_odom_data_[reproduce_sequence_indices_.front()].second;
  const double bag_speed = utils::calculate_speed_2d(bag_odom.twist.twist);
  const double bag_dist =
    utils::calculate_distance_2d(ego_pose.position, bag_odom.pose.pose.position);

  return (bag_speed > ego_speed * 2.0) && (bag_speed > 3.0) && (bag_dist > param_.search_radius);
}

void PerceptionReproducer::maybe_auto_tackle_stuck(
  const nav_msgs::msg::Odometry & ego_odom, bool & repeat_flag,
  const rclcpp::Time & current_timestamp)
{
  if (!get_parameter("auto_tackle_stuck").as_bool()) {
    stuck_since_.reset();
    return;
  }
  // Same gate as DLR engage_node: only act after Autoware has engaged.
  if (autoware_state_.load() != autoware_system_msgs::msg::AutowareState::DRIVING) {
    stuck_since_.reset();
    return;
  }

  const double ego_speed = utils::calculate_speed_2d(ego_odom.twist.twist);
  const bool ego_stopped = ego_speed < k_ego_stopped_speed_eps;
  if (!ego_stopped) {
    tried_expand_while_stopped_ = false;
    stuck_since_.reset();
    return;
  }

  if (initialize_in_flight_.load()) {
    return;
  }
  if (
    last_perturb_attempt_.has_value() &&
    (current_timestamp - last_perturb_attempt_.value()).seconds() < k_perturb_backoff_s) {
    return;
  }

  // ego is stopped; stuck means reproducer is in repeat
  if (!repeat_flag) {
    stuck_since_.reset();
    return;
  }

  if (!stuck_since_.has_value()) {
    stuck_since_ = current_timestamp;
    return;
  }

  const double stuck_duration_s = get_parameter("stuck_duration_s").as_double();
  const double stuck_s = (current_timestamp - stuck_since_.value()).seconds();
  if (stuck_s < stuck_duration_s) {
    return;
  }

  // First recovery: force rebuild with scaled search radius (does not change stored radius).
  if (!tried_expand_while_stopped_) {
    const double expand_radius_scale = get_parameter("expand_radius_scale").as_double();
    const double expanded_radius = expand_radius_scale * param_.search_radius;
    rebuild_reproduce_sequence(ego_odom.pose.pose, expanded_radius, current_timestamp);
    tried_expand_while_stopped_ = true;
    expand_count_.fetch_add(1);
    just_expanded_.store(true);
    stuck_since_.reset();
    repeat_flag = speed_gap_forces_repeat(ego_odom, ego_odom.pose.pose);

    RCLCPP_INFO(
      get_logger(),
      "auto-tackle expand: stuck for %.2fs, radius=%.3f -> %.3f (scale=%.3f), queue_size=%zu "
      "expand_count=%zu",
      stuck_s, param_.search_radius, expanded_radius, expand_radius_scale,
      reproduce_sequence_indices_.size(), expand_count_.load());
    return;
  }

  if (!last_published_timestamp_.has_value()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "auto-tackle perturb skipped: last_published_timestamp is not set yet");
    return;
  }

  const double perturb_distance_m = get_parameter("perturb_distance_m").as_double();
  const size_t start_idx =
    utils::get_nearest_index(rosbag_ego_odom_data_, last_published_timestamp_.value());
  const auto target_idx_opt = find_perturb_index_along_bag(start_idx, perturb_distance_m);
  if (!target_idx_opt.has_value()) {
    return;
  }

  const size_t target_idx = target_idx_opt.value();
  if (target_idx == start_idx) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "auto-tackle perturb skipped: target pose is same as current bag index %zu", start_idx);
    last_perturb_attempt_ = current_timestamp;
    stuck_since_.reset();
    return;
  }

  RCLCPP_INFO(
    get_logger(),
    "auto-tackle perturb: stuck for %.2fs, bag anchor %zu -> %zu (distance_param=%.3f m)", stuck_s,
    start_idx, target_idx, perturb_distance_m);

  if (!call_direct_initialize(target_idx)) {
    return;
  }

  last_perturb_attempt_ = current_timestamp;
  stuck_since_.reset();
  tried_expand_while_stopped_ = false;
}

bool PerceptionReproducer::call_direct_initialize(const size_t target_idx)
{
  // MultiThreadedExecutor: claim in-flight atomically to avoid duplicate sends.
  bool expected = false;
  if (!initialize_in_flight_.compare_exchange_strong(expected, true)) {
    return false;
  }

  if (!initialize_client_->service_is_ready()) {
    initialize_in_flight_.store(false);
    RCLCPP_ERROR(get_logger(), "auto-tackle perturb: /localization/initialize is not ready");
    return false;
  }

  const auto & pose = rosbag_ego_odom_data_[target_idx].second.pose.pose;
  auto request =
    std::make_shared<autoware_localization_msgs::srv::InitializeLocalization::Request>();
  request->method = autoware_localization_msgs::srv::InitializeLocalization::Request::DIRECT;
  request->pose_with_covariance = {utils::make_map_initial_pose(pose, this->get_clock()->now())};

  initialize_client_->async_send_request(
    request, [this, target_idx](
               rclcpp::Client<autoware_localization_msgs::srv::InitializeLocalization>::SharedFuture
                 future) { on_initialize_response(future, target_idx); });
  return true;
}

void PerceptionReproducer::on_initialize_response(
  rclcpp::Client<autoware_localization_msgs::srv::InitializeLocalization>::SharedFuture future,
  const size_t target_idx)
{
  initialize_in_flight_.store(false);

  try {
    const auto response = future.get();
    if (!response->status.success) {
      RCLCPP_ERROR(
        get_logger(), "auto-tackle perturb: /localization/initialize failed (code=%u)",
        response->status.code);
      return;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "auto-tackle perturb: initialize exception: %s", e.what());
    return;
  }

  pending_perturb_target_idx_.store(target_idx);
  pending_perturb_apply_.store(true);
  perturb_count_.fetch_add(1);
  just_perturbed_.store(true);
  RCLCPP_INFO(
    get_logger(), "auto-tackle perturb succeeded: target_idx=%zu perturb_count=%zu", target_idx,
    perturb_count_.load());
}

void PerceptionReproducer::flush_state_duration(const rclcpp::Time & now)
{
  if (!last_duration_stamp_.has_value() || last_duration_state_.empty()) {
    return;
  }

  const double dt = (now - last_duration_stamp_.value()).seconds();
  if (dt <= 0.0) {
    return;
  }

  if (last_duration_state_ == "normal") {
    normal_duration_total_s_ += dt;
  } else if (last_duration_state_ == "repeat") {
    repeat_duration_total_s_ += dt;
  }
  last_duration_stamp_ = now;
}

void PerceptionReproducer::update_state_duration(
  const std::string & next_state, const rclcpp::Time & now)
{
  flush_state_duration(now);
  last_duration_state_ = next_state;
  last_duration_stamp_ = now;
}

void PerceptionReproducer::publish_metrics(const std::string & state)
{
  tier4_metric_msgs::msg::MetricArray metrics_msg;
  metrics_msg.stamp = this->get_clock()->now();

  auto add_metric = [&metrics_msg](const std::string & name, const std::string & value) {
    tier4_metric_msgs::msg::Metric metric;
    metric.name = name;
    metric.value = value;
    metrics_msg.metric_array.push_back(metric);
  };

  add_metric("reproducer/state", state);
  add_metric("reproducer/perturb_count", std::to_string(perturb_count_.load()));
  add_metric("reproducer/expand_count", std::to_string(expand_count_.load()));
  metrics_pub_->publish(metrics_msg);
}

void PerceptionReproducer::dump_metrics_json()
{
  if (!get_parameter("output_metrics").as_bool()) {
    return;
  }

  try {
    nlohmann::json output_json;
    output_json["reproducer/perturb_count"] = perturb_count_.load();
    output_json["reproducer/expand_count"] = expand_count_.load();
    output_json["reproducer/normal_duration/total"] = normal_duration_total_s_;
    output_json["reproducer/repeat_duration/total"] = repeat_duration_total_s_;

    const std::string output_folder_str =
      rclcpp::get_logging_directory().string() + "/autoware_metrics";
    if (!std::filesystem::exists(output_folder_str)) {
      if (!std::filesystem::create_directories(output_folder_str)) {
        RCLCPP_ERROR(get_logger(), "Failed to create directories: %s", output_folder_str.c_str());
        return;
      }
    }

    const std::time_t now_time_t =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now_time_t);
#else
    localtime_r(&now_time_t, &local_time);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");

    const std::string output_file_str =
      output_folder_str + "/perception_reproducer-" + oss.str() + ".json";
    std::ofstream f(output_file_str);
    if (f.is_open()) {
      f << output_json.dump(4);
      f.close();
      RCLCPP_INFO(get_logger(), "Wrote metrics json: %s", output_file_str.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to open file: %s", output_file_str.c_str());
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to dump metrics json: %s", e.what());
  }
}

void PerceptionReproducer::on_timer()
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  const auto timer_start = std::chrono::high_resolution_clock::now();
  const auto current_timestamp = this->get_clock()->now();

  if (pending_perturb_apply_.exchange(false)) {
    const size_t target_idx = pending_perturb_target_idx_.load();
    if (target_idx < rosbag_ego_odom_data_.size()) {
      reset_reproduce_tracking(rosbag_ego_odom_data_[target_idx].first);
      RCLCPP_INFO(
        get_logger(), "auto-tackle perturb applied: bag timestamp=%.3f",
        last_published_timestamp_->seconds());
    }
  }

  // check if ego_odom is available
  const auto ego_odom_opt = get_latest_ego_odom();
  if (!ego_odom_opt.has_value()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "No ego odom found.");
    return;
  }

  const auto ego_odom = ego_odom_opt.value();
  const auto ego_pose = ego_odom.pose.pose;

  // calculate distance moved since last sequenced ego pose
  const double dist_moved =
    last_sequenced_ego_pose_.has_value()
      ? utils::calculate_distance_2d(ego_pose.position, last_sequenced_ego_pose_->position)
      : 999.0;

  // update the reproduce sequence if the distance moved is greater than the search radius
  if (dist_moved > param_.search_radius) {
    rebuild_reproduce_sequence(ego_pose, param_.search_radius, current_timestamp);
  }

  if (param_.verbose) {
    std::string indices_str;
    for (size_t i = 0; i < std::min<size_t>(20, reproduce_sequence_indices_.size()); ++i) {
      indices_str += std::to_string(reproduce_sequence_indices_[i]) + " ";
    }
    RCLCPP_INFO(get_logger(), "reproduce_sequence_indices: %s", indices_str.c_str());
  }

  bool repeat_flag = speed_gap_forces_repeat(ego_odom, ego_pose);

  // stuck: first expand search rebuild once, then perturb
  maybe_auto_tackle_stuck(ego_odom, repeat_flag, current_timestamp);
  // Only mark perturb after async initialize succeeds (count already incremented).
  const bool had_async_perturb = just_perturbed_.exchange(false);
  const bool had_expand = just_expanded_.exchange(false);

  const std::string duration_state = repeat_flag ? "repeat" : "normal";
  update_state_duration(duration_state, current_timestamp);

  const std::string publish_state =
    had_async_perturb ? "perturb" : (had_expand ? "expand" : duration_state);
  publish_metrics(publish_state);

  // publish messages
  const auto bag_timestamp = [&]() -> std::optional<rclcpp::Time> {
    if (!repeat_flag) {
      const size_t ego_odom_idx = reproduce_sequence_indices_.front();
      reproduce_sequence_indices_.pop_front();
      last_published_timestamp_ = rosbag_ego_odom_data_[ego_odom_idx].first;
      ego_odom_id2last_published_timestamp_[ego_odom_idx] = current_timestamp;
      cool_down_indices_.push_back(ego_odom_idx);
      return last_published_timestamp_;
    }

    if (!last_published_timestamp_.has_value()) {
      const auto idx = find_nearest_ego_odom_index(ego_pose);
      last_published_timestamp_ = rosbag_ego_odom_data_[idx].first;
      RCLCPP_INFO(
        get_logger(), "Seeded bag timestamp from ego pose (idx=%zu, t=%.3f)", idx,
        last_published_timestamp_->seconds());
    }

    return last_published_timestamp_;
  }();

  if (bag_timestamp.has_value()) {
    publish_topics_at_timestamp(
      bag_timestamp.value(), current_timestamp, param_.noise && repeat_flag);
  } else {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "No valid bag timestamp to publish.");
  }

  const auto timer_end = std::chrono::high_resolution_clock::now();
  const auto total_time =
    std::chrono::duration_cast<std::chrono::microseconds>(timer_end - timer_start).count() / 1000.0;

  if (param_.verbose) {
    RCLCPP_INFO(get_logger(), "on_timer processing time: %.3f ms", total_time);
  }
}

size_t PerceptionReproducer::find_nearest_ego_odom_index(
  const geometry_msgs::msg::Pose & ego_pose) const
{
  const double target_x = ego_pose.position.x;
  const double target_y = ego_pose.position.y;

  double min_dist_same_heading = std::numeric_limits<double>::max();
  double min_dist_any = std::numeric_limits<double>::max();
  size_t nearest_same_heading_idx = 0;
  size_t nearest_any_idx = 0;
  bool found_same_heading = false;

  for (size_t i = 0; i < rosbag_ego_odom_data_.size(); ++i) {
    const auto & bag_pose = rosbag_ego_odom_data_[i].second.pose.pose;
    const double dx = bag_pose.position.x - target_x;
    const double dy = bag_pose.position.y - target_y;
    const double dist_squared = dx * dx + dy * dy;

    if (dist_squared < min_dist_any) {
      min_dist_any = dist_squared;
      nearest_any_idx = i;
    }

    if (
      utils::absolute_yaw_difference(ego_pose.orientation, bag_pose.orientation) >
      k_max_heading_diff_for_nearby_rad) {
      continue;
    }

    if (dist_squared < min_dist_same_heading) {
      min_dist_same_heading = dist_squared;
      nearest_same_heading_idx = i;
      found_same_heading = true;
    }
  }

  return found_same_heading ? nearest_same_heading_idx : nearest_any_idx;
}

std::vector<size_t> PerceptionReproducer::find_nearby_ego_odom_indices(
  const std::vector<geometry_msgs::msg::Pose> & ego_poses, const double search_radius) const
{
  const double search_radius_squared = search_radius * search_radius;
  std::vector<size_t> nearby_indices;

  for (size_t i = 0; i < rosbag_ego_odom_data_.size(); ++i) {
    const auto & bag_pose = rosbag_ego_odom_data_[i].second.pose.pose;

    for (const auto & ego_pose : ego_poses) {
      if (
        utils::absolute_yaw_difference(ego_pose.orientation, bag_pose.orientation) >
        k_max_heading_diff_for_nearby_rad) {
        continue;
      }

      const double dx = bag_pose.position.x - ego_pose.position.x;
      const double dy = bag_pose.position.y - ego_pose.position.y;
      const double dist_squared = dx * dx + dy * dy;

      if (dist_squared <= search_radius_squared) {
        nearby_indices.push_back(i);
        break;  // found within radius, no need to check other ego_poses
      }
    }
  }

  return nearby_indices;
}

}  // namespace autoware::planning_debug_tools
