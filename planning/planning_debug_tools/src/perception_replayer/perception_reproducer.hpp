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

#ifndef PERCEPTION_REPLAYER__PERCEPTION_REPRODUCER_HPP_
#define PERCEPTION_REPLAYER__PERCEPTION_REPRODUCER_HPP_

#include "perception_replayer_common.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_localization_msgs/srv/initialize_localization.hpp>
#include <autoware_system_msgs/msg/autoware_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tier4_metric_msgs/msg/metric.hpp>
#include <tier4_metric_msgs/msg/metric_array.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::planning_debug_tools
{

struct PerceptionReproducerParam : public PerceptionReplayerCommonParam
{
  double search_radius;
  double reproduce_cool_down;
  bool noise;
  bool verbose;
  bool publish_route;
  bool auto_tackle_stuck{false};
  double stuck_duration_s{5.0};
  double expand_radius_scale{3.0};
  double perturb_distance_m{0.5};
  bool output_metrics{false};
};

class PerceptionReproducer : public PerceptionReplayerCommon
{
public:
  explicit PerceptionReproducer(
    const PerceptionReproducerParam & param, const rclcpp::NodeOptions & node_options);
  ~PerceptionReproducer() override;

private:
  void on_timer();
  void on_pose_reset(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void on_autoware_state(const autoware_system_msgs::msg::AutowareState::SharedPtr msg);

  // find nearest ego odom index by position (same-heading preferred)
  size_t find_nearest_ego_odom_index(const geometry_msgs::msg::Pose & ego_pose) const;

  // find nearby ego odom indices within search radius, excluding opposite heading
  std::vector<size_t> find_nearby_ego_odom_indices(
    const std::vector<geometry_msgs::msg::Pose> & ego_poses, const double search_radius) const;

  void reset_reproduce_tracking(const std::optional<rclcpp::Time> & new_bag_timestamp);

  void rebuild_reproduce_sequence(
    const geometry_msgs::msg::Pose & ego_pose, const double search_radius,
    const rclcpp::Time & current_timestamp);

  std::optional<size_t> find_perturb_index_along_bag(
    const size_t start_idx, const double distance_m) const;
  void maybe_auto_tackle_stuck(
    const nav_msgs::msg::Odometry & ego_odom, bool & repeat_flag,
    const rclcpp::Time & current_timestamp);
  bool speed_gap_forces_repeat(
    const nav_msgs::msg::Odometry & ego_odom, const geometry_msgs::msg::Pose & ego_pose) const;
  bool call_direct_initialize(const size_t target_idx);
  void on_initialize_response(
    rclcpp::Client<autoware_localization_msgs::srv::InitializeLocalization>::SharedFuture future,
    const size_t target_idx);

  void flush_state_duration(const rclcpp::Time & now);
  void update_state_duration(const std::string & next_state, const rclcpp::Time & now);
  void publish_metrics(const std::string & state);
  void dump_metrics_json();

private:
  // parameters
  const PerceptionReproducerParam param_;

  // timer: MutuallyExclusive avoids overlapping on_timer; state_mutex_ also guards pose-reset
  rclcpp::CallbackGroup::SharedPtr timer_callback_group_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::mutex state_mutex_;

  // subscription
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pos_;
  rclcpp::Subscription<autoware_system_msgs::msg::AutowareState>::SharedPtr sub_autoware_state_;
  std::atomic<uint8_t> autoware_state_{autoware_system_msgs::msg::AutowareState::INITIALIZING};

  // localization initialize
  rclcpp::Client<autoware_localization_msgs::srv::InitializeLocalization>::SharedPtr
    initialize_client_;
  std::atomic<bool> initialize_in_flight_{false};
  std::optional<rclcpp::Time> stuck_since_;
  std::optional<rclcpp::Time> last_perturb_attempt_;
  bool tried_expand_while_stopped_{false};

  // metrics
  rclcpp::Publisher<tier4_metric_msgs::msg::MetricArray>::SharedPtr metrics_pub_;
  std::string last_duration_state_;
  std::optional<rclcpp::Time> last_duration_stamp_;
  double normal_duration_total_s_{0.0};
  double repeat_duration_total_s_{0.0};
  std::atomic<size_t> perturb_count_{0};
  std::atomic<size_t> expand_count_{0};
  std::atomic<bool> just_perturbed_{false};
  std::atomic<bool> just_expanded_{false};
  std::atomic<bool> pending_perturb_apply_{false};
  std::atomic<size_t> pending_perturb_target_idx_{0};

  // state management
  std::deque<size_t> reproduce_sequence_indices_;
  std::deque<size_t> cool_down_indices_;
  std::unordered_map<size_t, rclcpp::Time> ego_odom_id2last_published_timestamp_;
  std::optional<geometry_msgs::msg::Pose> last_sequenced_ego_pose_;
  std::optional<rclcpp::Time> last_published_timestamp_;
};

}  // namespace autoware::planning_debug_tools

#endif  // PERCEPTION_REPLAYER__PERCEPTION_REPRODUCER_HPP_
