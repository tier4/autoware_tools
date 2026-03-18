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

#ifndef TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_HPP_
#define TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_HPP_

#include "../perception_replayer/time_manager_widget.hpp"
#include "trajectory_replayer_common.hpp"

#include <rclcpp/rclcpp.hpp>

namespace autoware::planning_debug_tools::trajectory_replayer
{

class TrajectoryReplayer : public TrajectoryReplayerCommon
{
public:
  explicit TrajectoryReplayer(
    const TrajectoryReplayerParam & param, const rclcpp::NodeOptions & node_options);

private:
  void on_set_rate(const QString & rate_text);
  void on_timer();

  TimeManagerWidget * widget_;
  rclcpp::TimerBase::SharedPtr timer_;
  double rate_ = 1.0;
  static constexpr double delta_time_ = 0.05;
  bool publish_trajectories_next_ = false;
};

}  // namespace autoware::planning_debug_tools::trajectory_replayer

#endif  // TRAJECTORY_REPLAYER__TRAJECTORY_REPLAYER_HPP_
