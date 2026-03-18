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

#pragma once

#include <autoware_internal_planning_msgs/msg/candidate_trajectories.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace autoware::planning_debug_tools::trajectory_replayer
{

using CandidateTrajectories = autoware_internal_planning_msgs::msg::CandidateTrajectories;
using Odometry = nav_msgs::msg::Odometry;
using AccelWithCovarianceStamped = geometry_msgs::msg::AccelWithCovarianceStamped;
using PredictedObjects = autoware_perception_msgs::msg::PredictedObjects;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
using TFMessage = tf2_msgs::msg::TFMessage;

}  // namespace autoware::planning_debug_tools::trajectory_replayer
