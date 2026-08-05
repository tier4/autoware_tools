// Copyright 2025 The Autoware Foundation.

#include "autoware/perfect_tracker/perfect_tracker_node.hpp"

#include <autoware/lanelet2_utils/conversion.hpp>
#include <autoware/motion_utils/trajectory/trajectory.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_utils_geometry/geometry.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <lanelet2_core/geometry/LaneletMap.h>  // Required for Fast Search
#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::simulator::perfect_tracker
{

using rclcpp::QoS;
using std::placeholders::_1;
using std::placeholders::_2;

PerfectTrackerNode::PerfectTrackerNode(const rclcpp::NodeOptions & options)
: Node("perfect_tracker_node", options), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
{
  simulated_frame_id_ = declare_parameter("simulated_frame_id", "base_link");
  origin_frame_id_ = declare_parameter("origin_frame_id", "map");
  n_delay_steps_ = declare_parameter("n_delay_steps", 0);
  tracker_mode_ = declare_parameter<std::string>("tracker_mode", "snap");
  replan_interval_ = declare_parameter("replan_interval", 1);
  simulate_motion_ = declare_parameter("initial_engage_state", false);

  // --- Subscribers ---
  sub_map_ = create_subscription<autoware_map_msgs::msg::LaneletMapBin>(
    "/map/vector_map", rclcpp::QoS(10).transient_local(),
    std::bind(&PerfectTrackerNode::on_map, this, _1));

  sub_trajectory_ = create_subscription<autoware_planning_msgs::msg::Trajectory>(
    "input/trajectory", QoS{1}, std::bind(&PerfectTrackerNode::on_trajectory, this, _1));

  sub_turn_indicators_cmd_ = create_subscription<autoware_vehicle_msgs::msg::TurnIndicatorsCommand>(
    "input/turn_indicators_command", QoS{1},
    std::bind(&PerfectTrackerNode::on_turn_indicators_cmd, this, _1));

  sub_hazard_lights_cmd_ = create_subscription<autoware_vehicle_msgs::msg::HazardLightsCommand>(
    "input/hazard_lights_command", QoS{1},
    std::bind(&PerfectTrackerNode::on_hazard_lights_cmd, this, _1));

  sub_init_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "input/initialpose", QoS{1}, std::bind(&PerfectTrackerNode::on_initialpose, this, _1));

  sub_init_twist_ = create_subscription<geometry_msgs::msg::TwistStamped>(
    "input/initialtwist", QoS{1}, std::bind(&PerfectTrackerNode::on_initialtwist, this, _1));

  // NOTE: In the planning simulator, /vehicle/engage has no publisher; the live engage signal
  // is /autoware/engage (published by operation_mode_transition_manager on engage transitions).
  sub_engage_ = create_subscription<autoware_vehicle_msgs::msg::Engage>(
    "/autoware/engage", QoS{1}, std::bind(&PerfectTrackerNode::on_engage, this, _1));

  // --- Publishers ---
  pub_velocity_ =
    create_publisher<autoware_vehicle_msgs::msg::VelocityReport>("output/twist", QoS{1});
  pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("output/odometry", QoS{1});
  pub_steer_ =
    create_publisher<autoware_vehicle_msgs::msg::SteeringReport>("output/steering", QoS{1});
  pub_acc_ =
    create_publisher<geometry_msgs::msg::AccelWithCovarianceStamped>("output/acceleration", QoS{1});
  pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("output/imu", QoS{1});
  pub_control_mode_report_ = create_publisher<autoware_vehicle_msgs::msg::ControlModeReport>(
    "output/control_mode_report", QoS{1});
  pub_gear_report_ =
    create_publisher<autoware_vehicle_msgs::msg::GearReport>("output/gear_report", QoS{1});
  pub_turn_indicators_report_ = create_publisher<autoware_vehicle_msgs::msg::TurnIndicatorsReport>(
    "output/turn_indicators_report", QoS{1});
  pub_hazard_lights_report_ = create_publisher<autoware_vehicle_msgs::msg::HazardLightsReport>(
    "output/hazard_lights_report", QoS{1});
  pub_current_pose_ =
    create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("output/pose", QoS{1});
  pub_tf_ = create_publisher<tf2_msgs::msg::TFMessage>("/tf", QoS{1});

  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  srv_mode_req_ = create_service<autoware_vehicle_msgs::srv::ControlModeCommand>(
    "input/control_mode_request",
    std::bind(&PerfectTrackerNode::on_control_mode_request, this, _1, _2));

  tier4_api_utils::ServiceProxyNodeInterface proxy(this);
  group_api_service_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  srv_set_pose_ = proxy.create_service<tier4_external_api_msgs::srv::InitializePose>(
    "/api/simulator/set/pose", std::bind(&PerfectTrackerNode::on_set_pose, this, _1, _2),
    rmw_qos_profile_services_default, group_api_service_);

  const auto period = std::chrono::milliseconds(100);
  on_timer_ =
    rclcpp::create_timer(this, get_clock(), period, std::bind(&PerfectTrackerNode::on_timer, this));

  current_control_mode_.mode = autoware_vehicle_msgs::msg::ControlModeReport::AUTONOMOUS;
  current_odometry_.pose.pose.orientation.w = 1.0;
}

void PerfectTrackerNode::on_map(const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg)
{
  lanelet_map_ptr_ = autoware::experimental::lanelet2_utils::from_autoware_map_msgs(*msg);

  has_map_ = true;
  RCLCPP_INFO(get_logger(), "Map loaded successfully.");
}

void PerfectTrackerNode::on_trajectory(
  const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg)
{
  trajectory_history_.push_front(msg);
  while (trajectory_history_.size() > 20) {
    trajectory_history_.pop_back();
  }
}

void PerfectTrackerNode::on_turn_indicators_cmd(
  const autoware_vehicle_msgs::msg::TurnIndicatorsCommand::ConstSharedPtr msg)
{
  current_turn_indicators_cmd_ptr_ = msg;
}

void PerfectTrackerNode::on_hazard_lights_cmd(
  const autoware_vehicle_msgs::msg::HazardLightsCommand::ConstSharedPtr msg)
{
  current_hazard_lights_cmd_ptr_ = msg;
}

void PerfectTrackerNode::on_gear_cmd(
  const autoware_vehicle_msgs::msg::GearCommand::ConstSharedPtr msg)
{
  current_gear_cmd_ = *msg;
}

void PerfectTrackerNode::on_initialpose(
  const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg)
{
  geometry_msgs::msg::Twist zero_twist;
  geometry_msgs::msg::Pose pose = msg->pose.pose;
  if (msg->header.frame_id != origin_frame_id_) {
    try {
      auto transform = get_transform_msg(origin_frame_id_, msg->header.frame_id);
      pose.position.x += transform.transform.translation.x;
      pose.position.y += transform.transform.translation.y;
      pose.position.z += transform.transform.translation.z;
    } catch (const std::exception &) {
    }
  }
  set_initial_state(pose, zero_twist);
}

void PerfectTrackerNode::on_initialtwist(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg)
{
  if (is_initialized_) {
    current_odometry_.twist.twist = msg->twist;
  }
}

void PerfectTrackerNode::on_engage(const autoware_vehicle_msgs::msg::Engage::ConstSharedPtr msg)
{
  simulate_motion_ = msg->engage;
}

void PerfectTrackerNode::on_set_pose(
  const tier4_external_api_msgs::srv::InitializePose::Request::ConstSharedPtr request,
  const tier4_external_api_msgs::srv::InitializePose::Response::SharedPtr response)
{
  geometry_msgs::msg::Twist zero_twist;
  set_initial_state(request->pose.pose.pose, zero_twist);
  response->status = tier4_api_utils::response_success();
}

void PerfectTrackerNode::on_control_mode_request(
  const autoware_vehicle_msgs::srv::ControlModeCommand::Request::ConstSharedPtr request,
  const autoware_vehicle_msgs::srv::ControlModeCommand::Response::SharedPtr response)
{
  current_control_mode_.mode = request->mode;
  response->success = true;
}

void PerfectTrackerNode::set_initial_state(
  const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Twist & twist)
{
  current_odometry_.pose.pose = pose;
  current_odometry_.twist.twist = twist;
  is_initialized_ = true;
  last_closest_idx_ = 0;
  cached_plan_ = nullptr;
  plan_cursor_ = 0;
  step_counter_ = 0;
}

void PerfectTrackerNode::on_timer()
{
  if (!is_initialized_) {
    pub_control_mode_report_->publish(current_control_mode_);
    return;
  }

  // 1. [SPS Logic] Calculate Pitch from Map (Using Fast Search!)
  double ego_pitch = 0.0;
  if (has_map_) {
    ego_pitch = calculate_ego_pitch();
  }

  // 2. Select Trajectory
  current_trajectory_ptr_ = nullptr;
  if (!trajectory_history_.empty()) {
    size_t trajectory_index = static_cast<size_t>(n_delay_steps_);
    if (trajectory_history_.size() <= trajectory_index) {
      current_trajectory_ptr_ = trajectory_history_.back();
    } else {
      current_trajectory_ptr_ = trajectory_history_[trajectory_index];
    }
  }

  // 3. Main Update
  // Hold the vehicle in place until it is engaged (simulate_motion_).
  if (
    simulate_motion_ &&
    current_control_mode_.mode == autoware_vehicle_msgs::msg::ControlModeReport::AUTONOMOUS &&
    current_trajectory_ptr_) {
    // Auto-Gear Logic
    if (!current_trajectory_ptr_->points.empty()) {
      size_t safe_idx = std::min(last_closest_idx_, current_trajectory_ptr_->points.size() - 1);
      double v_target = current_trajectory_ptr_->points[safe_idx].longitudinal_velocity_mps;
      current_gear_cmd_.command = (v_target < -0.01)
                                    ? autoware_vehicle_msgs::msg::GearCommand::REVERSE
                                    : autoware_vehicle_msgs::msg::GearCommand::DRIVE;
    }

    // Advance the ego using the selected algorithm
    if (tracker_mode_ == "physics") {
      update_state_from_trajectory_and_map(ego_pitch);
    } else {  // "snap"
      update_state_snap();
    }

  } else {
    // Force Stop
    current_odometry_.twist.twist.linear.x = 0.0;
    current_odometry_.twist.twist.angular.z = 0.0;
    current_gear_cmd_.command = autoware_vehicle_msgs::msg::GearCommand::PARK;
  }

  // 4. Publish Everything
  rclcpp::Time now = get_clock()->now();

  current_odometry_.header.stamp = now;
  current_odometry_.header.frame_id = origin_frame_id_;
  current_odometry_.child_frame_id = simulated_frame_id_;
  publish_odometry(current_odometry_);
  publish_tf(current_odometry_);

  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  pose_msg.header = current_odometry_.header;
  pose_msg.pose = current_odometry_.pose;
  pub_current_pose_->publish(pose_msg);

  autoware_vehicle_msgs::msg::VelocityReport vel_msg;
  vel_msg.header.stamp = now;
  vel_msg.header.frame_id = simulated_frame_id_;
  vel_msg.longitudinal_velocity = current_odometry_.twist.twist.linear.x;
  vel_msg.heading_rate = current_odometry_.twist.twist.angular.z;
  publish_velocity(vel_msg);

  autoware_vehicle_msgs::msg::SteeringReport steer_msg;
  steer_msg.stamp = now;
  steer_msg.steering_tire_angle = current_target_steering_;
  publish_steering(steer_msg);

  autoware_vehicle_msgs::msg::GearReport gear_msg;
  gear_msg.stamp = now;
  gear_msg.report = current_gear_cmd_.command;
  pub_gear_report_->publish(gear_msg);

  current_control_mode_.stamp = now;
  pub_control_mode_report_->publish(current_control_mode_);

  autoware_vehicle_msgs::msg::TurnIndicatorsReport turn_msg;
  turn_msg.stamp = now;
  turn_msg.report = current_turn_indicators_cmd_ptr_
                      ? current_turn_indicators_cmd_ptr_->command
                      : autoware_vehicle_msgs::msg::TurnIndicatorsReport::DISABLE;
  pub_turn_indicators_report_->publish(turn_msg);

  autoware_vehicle_msgs::msg::HazardLightsReport hazard_msg;
  hazard_msg.stamp = now;
  hazard_msg.report = current_hazard_lights_cmd_ptr_
                        ? current_hazard_lights_cmd_ptr_->command
                        : autoware_vehicle_msgs::msg::HazardLightsReport::DISABLE;
  pub_hazard_lights_report_->publish(hazard_msg);

  publish_acceleration();
  publish_imu();
}

// ==========================================================================================
// LOGIC: MIMIC SPS EXACTLY (Hybrid Approach) with FAST Search
// ==========================================================================================

// Helper: Extract points from lanelet
namespace
{
std::vector<geometry_msgs::msg::Point> convert_centerline_to_points(
  const lanelet::ConstLanelet & lanelet)
{
  std::vector<geometry_msgs::msg::Point> centerline_points;
  for (const auto & point : lanelet.centerline()) {
    geometry_msgs::msg::Point center_point;
    center_point.x = point.basicPoint().x();
    center_point.y = point.basicPoint().y();
    center_point.z = point.basicPoint().z();
    centerline_points.push_back(center_point);
  }
  return centerline_points;
}
}  // namespace

void PerfectTrackerNode::update_state_from_trajectory_and_map(
  [[maybe_unused]] const double ego_pitch)
{
  if (!current_trajectory_ptr_ || current_trajectory_ptr_->points.empty()) return;

  const double dt = 0.100;

  // --------------------------------------------------------
  // STEP 1: Find Target Velocity & Kinematics (Lookahead)
  // --------------------------------------------------------
  double cx = current_odometry_.pose.pose.position.x;
  double cy = current_odometry_.pose.pose.position.y;

  // Find closest point for target velocity
  double min_dist = std::numeric_limits<double>::max();
  size_t closest_idx = 0;
  for (size_t i = 0; i < current_trajectory_ptr_->points.size(); ++i) {
    const auto & p = current_trajectory_ptr_->points[i];
    double d2 = (cx - p.pose.position.x) * (cx - p.pose.position.x) +
                (cy - p.pose.position.y) * (cy - p.pose.position.y);
    if (d2 < min_dist) {
      min_dist = d2;
      closest_idx = i;
    }
  }
  last_closest_idx_ = closest_idx;
  const auto & target_pt = current_trajectory_ptr_->points[closest_idx];

  // Set dynamics targets
  current_odometry_.twist.twist.linear.x = target_pt.longitudinal_velocity_mps;
  current_odometry_.twist.twist.angular.z = target_pt.heading_rate_rps;
  current_target_steering_ = target_pt.front_wheel_angle_rad;
  current_acceleration_.accel.accel.linear.x = target_pt.acceleration_mps2;
  current_acceleration_.header.stamp = get_clock()->now();

  // --------------------------------------------------------
  // STEP 2: Move Forward (Euler Integration) - X, Y, Yaw
  // --------------------------------------------------------
  // Note: We use the OLD orientation to move forward
  double current_yaw = tf2::getYaw(current_odometry_.pose.pose.orientation);
  double v_target = target_pt.longitudinal_velocity_mps;

  // Update Position (X, Y)
  current_odometry_.pose.pose.position.x = cx + (v_target * std::cos(current_yaw) * dt);
  current_odometry_.pose.pose.position.y = cy + (v_target * std::sin(current_yaw) * dt);

  // Update Yaw (Heading) - Simple integration or snap to trajectory
  // SPS snaps to trajectory yaw, so we do that too:
  double next_yaw = tf2::getYaw(target_pt.pose.orientation);

  // --------------------------------------------------------
  // STEP 3: Calculate Terrain (Z & Pitch) at NEW Position
  // --------------------------------------------------------
  // Now that X/Y are updated, we find the ground truth for THIS location.

  // A. Interpolate Z (Height)
  // We pass the *new* odometry to the helper
  current_odometry_.pose.pose.position.z = get_z_pose_from_trajectory(
    current_odometry_.pose.pose.position.x, current_odometry_.pose.pose.position.y,
    current_odometry_  // acts as prev_odometry fallback
  );

  // B. Calculate Pitch from Map (using NEW X/Y)
  double new_pitch = 0.0;
  if (has_map_) {
    new_pitch =
      calculate_ego_pitch();  // This function reads current_odometry_, which is now updated
  }

  // --------------------------------------------------------
  // STEP 4: Apply Final Orientation
  // --------------------------------------------------------
  current_odometry_.pose.pose.orientation =
    autoware_utils_geometry::create_quaternion_from_rpy(0.0, new_pitch, next_yaw);
}

void PerfectTrackerNode::update_state_snap()
{
  // Snap the ego onto the model's predicted pose by plan index. Adopt a fresh plan only at a
  // re-plan boundary (every replan_interval steps), restarting the cursor at point[0]. Otherwise
  // (holding, or a boundary where the plan has NOT actually updated) advance the cursor to the next
  // predicted point so the ego keeps moving instead of re-snapping onto one point.
  if (!current_trajectory_ptr_ || current_trajectory_ptr_->points.empty()) return;

  const double dt = 0.100;
  const size_t interval = static_cast<size_t>(std::max(1, replan_interval_));
  const bool at_replan_boundary = (step_counter_ % interval == 0);
  const bool plan_updated = (cached_plan_ != current_trajectory_ptr_);
  ++step_counter_;

  if (at_replan_boundary && plan_updated) {
    cached_plan_ =
      current_trajectory_ptr_;  // fresh plan at a re-plan boundary -> follow from start
    plan_cursor_ = 0;
  } else if (cached_plan_ && !cached_plan_->points.empty()) {
    ++plan_cursor_;  // holding, or boundary with no update -> step to the next predicted point
  } else {
    cached_plan_ = current_trajectory_ptr_;  // no plan yet -> bootstrap
    plan_cursor_ = 0;
  }

  const auto & pts = cached_plan_->points;
  const size_t idx = std::min(plan_cursor_, pts.size() - 1);
  last_closest_idx_ = idx;
  const auto & target_pt = pts[idx];

  const double cx = current_odometry_.pose.pose.position.x;
  const double cy = current_odometry_.pose.pose.position.y;
  const double cur_yaw = tf2::getYaw(current_odometry_.pose.pose.orientation);
  const double cur_speed = current_odometry_.twist.twist.linear.x;

  // Snap the ego DIRECTLY onto the predicted world pose.
  const double new_x = target_pt.pose.position.x;
  const double new_y = target_pt.pose.position.y;
  const double new_yaw = tf2::getYaw(target_pt.pose.orientation);
  const double new_speed = std::hypot(new_x - cx, new_y - cy) / dt;

  // Position: X/Y snapped; Z keeps the existing trajectory-interpolation behavior.
  current_odometry_.pose.pose.position.x = new_x;
  current_odometry_.pose.pose.position.y = new_y;
  current_odometry_.pose.pose.position.z =
    get_z_pose_from_trajectory(new_x, new_y, current_odometry_);

  // Orientation: yaw from the plan, pitch from the map (reads current_odometry_ at the new X/Y).
  const double new_pitch = has_map_ ? calculate_ego_pitch() : 0.0;
  current_odometry_.pose.pose.orientation =
    autoware_utils_geometry::create_quaternion_from_rpy(0.0, new_pitch, new_yaw);

  // Twist / acceleration / steering, reported consistently with the Python ego dynamics.
  const double dh = std::atan2(std::sin(new_yaw - cur_yaw), std::cos(new_yaw - cur_yaw));
  current_odometry_.twist.twist.linear.x = new_speed;
  current_odometry_.twist.twist.angular.z = dh / dt;
  current_acceleration_.accel.accel.linear.x = (new_speed - cur_speed) / dt;
  current_acceleration_.header.stamp = get_clock()->now();
  current_target_steering_ = 0.0;
}

double PerfectTrackerNode::calculate_ego_pitch() const
{
  if (!lanelet_map_ptr_) return 0.0;

  lanelet::BasicPoint2d search_point(
    current_odometry_.pose.pose.position.x, current_odometry_.pose.pose.position.y);

  // [FAST SEARCH] Use the R-Tree from the map object directly
  // This finds the 1 nearest lanelet in O(log N) time, preventing lag.

  auto nearest_lanelets =
    lanelet::geometry::findNearest(lanelet_map_ptr_->laneletLayer, search_point, 1);

  if (nearest_lanelets.empty()) return 0.0;

  lanelet::ConstLanelet ego_lanelet = nearest_lanelets[0].second;

  const auto centerline_points = convert_centerline_to_points(ego_lanelet);
  if (centerline_points.size() < 2) return 0.0;

  const size_t ego_seg_idx = autoware::motion_utils::findNearestSegmentIndex(
    centerline_points, current_odometry_.pose.pose.position);
  const auto & prev_point = centerline_points.at(ego_seg_idx);
  const auto & next_point = centerline_points.at(ego_seg_idx + 1);

  // [SPS MIMIC] Exact Math from SimplePlanningSimulator
  const double road_yaw = std::atan2(next_point.y - prev_point.y, next_point.x - prev_point.x);
  const double car_yaw = tf2::getYaw(current_odometry_.pose.pose.orientation);
  const double yaw_diff = car_yaw - road_yaw;

  const double diff_z = next_point.z - prev_point.z;
  const double diff_xy = std::hypot(next_point.x - prev_point.x, next_point.y - prev_point.y);
  const double projected_run = diff_xy / std::cos(yaw_diff);
  const bool reverse_sign = std::cos(yaw_diff) < 0.0;

  return reverse_sign ? -std::atan2(-diff_z, -projected_run) : -std::atan2(diff_z, projected_run);
}

double PerfectTrackerNode::get_z_pose_from_trajectory(
  const double x, const double y, const nav_msgs::msg::Odometry & prev_odometry)
{
  // 1. Safety Checks
  if (!current_trajectory_ptr_ || current_trajectory_ptr_->points.size() < 2) {
    return prev_odometry.pose.pose.position.z;
  }

  // 2. Find the Closest Point
  // --------------------------------------------------------
  double min_dist_sq = std::numeric_limits<double>::max();
  size_t closest_idx = 0;

  for (size_t i = 0; i < current_trajectory_ptr_->points.size(); ++i) {
    const auto & p = current_trajectory_ptr_->points[i];
    double d2 = (p.pose.position.x - x) * (p.pose.position.x - x) +
                (p.pose.position.y - y) * (p.pose.position.y - y);
    if (d2 < min_dist_sq) {
      min_dist_sq = d2;
      closest_idx = i;
    }
  }

  // 3. Identify the Segment (Previous & Next Points)
  // --------------------------------------------------------
  // We need to know if we are 'ahead' or 'behind' the closest point
  // to pick the correct neighbor for interpolation.
  size_t prev_idx = closest_idx;
  size_t next_idx = closest_idx + 1;

  // If closest is the last point, look backward
  if (next_idx >= current_trajectory_ptr_->points.size()) {
    next_idx = closest_idx;
    prev_idx = (closest_idx > 0) ? closest_idx - 1 : 0;
  } else if (closest_idx > 0) {
    // If we are not at the end, check if we should look backward instead
    const auto & p_curr = current_trajectory_ptr_->points[closest_idx];
    const auto & p_next = current_trajectory_ptr_->points[next_idx];

    // Project vector to see where we lie
    double dx = p_next.pose.position.x - p_curr.pose.position.x;
    double dy = p_next.pose.position.y - p_curr.pose.position.y;
    double v_x = x - p_curr.pose.position.x;
    double v_y = y - p_curr.pose.position.y;

    // If dot product is negative, we are 'behind' the current point
    if ((dx * v_x + dy * v_y) < 0) {
      next_idx = closest_idx;
      prev_idx = closest_idx - 1;
    }
  }

  // 4. Linear Interpolation (Lerp)
  // --------------------------------------------------------
  const auto & p1 = current_trajectory_ptr_->points[prev_idx];
  const auto & p2 = current_trajectory_ptr_->points[next_idx];

  double dx = p2.pose.position.x - p1.pose.position.x;
  double dy = p2.pose.position.y - p1.pose.position.y;
  double dz = p2.pose.position.z - p1.pose.position.z;
  double len_sq = dx * dx + dy * dy;

  // Default to nearest Z if segment is too short (vertical line or duplicate points)
  if (len_sq < 1e-6) {
    return p1.pose.position.z;
  }

  // Calculate ratio t (0.0 to 1.0) along the segment
  double v_x = x - p1.pose.position.x;
  double v_y = y - p1.pose.position.y;
  double t = (v_x * dx + v_y * dy) / len_sq;
  t = std::clamp(t, 0.0, 1.0);

  // Return interpolated Z
  return p1.pose.position.z + t * dz;
}

geometry_msgs::msg::TransformStamped PerfectTrackerNode::get_transform_msg(
  const std::string parent_frame, const std::string child_frame)
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_.lookupTransform(parent_frame, child_frame, tf2::TimePointZero);
  } catch (tf2::TransformException &) {
  }
  return transform;
}

void PerfectTrackerNode::publish_velocity(
  const autoware_vehicle_msgs::msg::VelocityReport & velocity)
{
  pub_velocity_->publish(velocity);
}

void PerfectTrackerNode::publish_odometry(const nav_msgs::msg::Odometry & odometry)
{
  pub_odom_->publish(odometry);
}

void PerfectTrackerNode::publish_steering(const autoware_vehicle_msgs::msg::SteeringReport & steer)
{
  pub_steer_->publish(steer);
}

void PerfectTrackerNode::publish_acceleration()
{
  pub_acc_->publish(current_acceleration_);
}

void PerfectTrackerNode::publish_imu()
{
  sensor_msgs::msg::Imu imu;
  imu.header.frame_id = simulated_frame_id_;
  imu.header.stamp = get_clock()->now();
  imu.orientation = current_odometry_.pose.pose.orientation;
  pub_imu_->publish(imu);
}

void PerfectTrackerNode::publish_tf(const nav_msgs::msg::Odometry & odometry)
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = get_clock()->now();
  tf.header.frame_id = origin_frame_id_;
  tf.child_frame_id = simulated_frame_id_;
  tf.transform.translation.x = odometry.pose.pose.position.x;
  tf.transform.translation.y = odometry.pose.pose.position.y;
  tf.transform.translation.z = odometry.pose.pose.position.z;
  tf.transform.rotation = odometry.pose.pose.orientation;

  tf2_msgs::msg::TFMessage tf_msg{};
  tf_msg.transforms.emplace_back(std::move(tf));
  pub_tf_->publish(tf_msg);
}

}  // namespace autoware::simulator::perfect_tracker

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::simulator::perfect_tracker::PerfectTrackerNode)
