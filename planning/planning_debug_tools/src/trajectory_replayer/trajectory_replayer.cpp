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

#include "trajectory_replayer.hpp"

namespace autoware::planning_debug_tools::trajectory_replayer
{

TrajectoryReplayer::TrajectoryReplayer(
  const TrajectoryReplayerParam & param, const rclcpp::NodeOptions & node_options)
: TrajectoryReplayerCommon(param, "trajectory_replayer", node_options)
{
  widget_ = new TimeManagerWidget(get_bag_start_time(), get_bag_end_timestamp());
  widget_->setObjectName("TrajectoryReplayer");
  widget_->setWindowTitle("Trajectory Replayer");
  widget_->show();

  for (auto * rate_button : widget_->rate_buttons) {
    QObject::connect(rate_button, &QPushButton::clicked, [this, rate_button]() {
      on_set_rate(rate_button->text());
    });
  }

  QObject::connect(widget_, &TimeManagerWidget::windowClosed, []() { rclcpp::shutdown(); });

  timer_ = rclcpp::create_timer(
    this, get_clock(), std::chrono::milliseconds(static_cast<int>(delta_time_ * 1000)),
    std::bind(&TrajectoryReplayer::on_timer, this), callback_group_);

  RCLCPP_INFO(get_logger(), "Start timer callback");
}

void TrajectoryReplayer::on_timer()
{
  const auto current_timestamp = this->get_clock()->now();

  rclcpp::Time bag_timestamp = widget_->get_slider_timestamp();

  if (!widget_->pause_button->isChecked() && !publish_trajectories_next_) {
    bag_timestamp += rclcpp::Duration::from_seconds(delta_time_ * 2.0 * rate_);

    const auto bag_end_timestamp = get_bag_end_timestamp();
    if (bag_timestamp >= bag_end_timestamp) {
      bag_timestamp = bag_end_timestamp;
    }

    widget_->set_slider_timestamp(bag_timestamp);
  }

  if (publish_trajectories_next_) {
    publish_trajectory_input(bag_timestamp, current_timestamp);
  } else {
    publish_sensor_data(bag_timestamp, current_timestamp);
  }
  publish_trajectories_next_ = !publish_trajectories_next_;
}

void TrajectoryReplayer::on_set_rate(const QString & rate_text)
{
  bool ok;
  const double new_rate = rate_text.toDouble(&ok);
  if (ok) {
    rate_ = new_rate;
  }
}

}  // namespace autoware::planning_debug_tools::trajectory_replayer
