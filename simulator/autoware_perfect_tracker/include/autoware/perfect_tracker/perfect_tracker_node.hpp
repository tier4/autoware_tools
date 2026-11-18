// Copyright 2025 The Autoware Foundation.

#ifndef AUTOWARE__PERFECT_TRACKER__PERFECT_TRACKER_NODE_HPP_
#define AUTOWARE__PERFECT_TRACKER__PERFECT_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <tier4_api_utils/tier4_api_utils.hpp>

#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_vehicle_msgs/msg/control_mode_report.hpp>
#include <autoware_vehicle_msgs/msg/engage.hpp>
#include <autoware_vehicle_msgs/msg/gear_command.hpp>
#include <autoware_vehicle_msgs/msg/gear_report.hpp>
#include <autoware_vehicle_msgs/msg/hazard_lights_command.hpp>
#include <autoware_vehicle_msgs/msg/hazard_lights_report.hpp>
#include <autoware_vehicle_msgs/msg/steering_report.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_command.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <autoware_vehicle_msgs/msg/velocity_report.hpp>
#include <autoware_vehicle_msgs/srv/control_mode_command.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tier4_external_api_msgs/srv/initialize_pose.hpp>

// [SPS MIMIC] Map Headers
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace autoware::simulator::perfect_tracker
{

class PerfectTrackerNode : public rclcpp::Node
{
public:
  explicit PerfectTrackerNode(const rclcpp::NodeOptions & options);

private:
  // --- Publishers ---
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::VelocityReport>::SharedPtr pub_velocity_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::SteeringReport>::SharedPtr pub_steer_;
  rclcpp::Publisher<geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr pub_acc_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::ControlModeReport>::SharedPtr
    pub_control_mode_report_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::GearReport>::SharedPtr pub_gear_report_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::TurnIndicatorsReport>::SharedPtr
    pub_turn_indicators_report_;
  rclcpp::Publisher<autoware_vehicle_msgs::msg::HazardLightsReport>::SharedPtr
    pub_hazard_lights_report_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_current_pose_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr pub_tf_;

  // --- Subscribers ---
  rclcpp::Subscription<autoware_map_msgs::msg::LaneletMapBin>::SharedPtr sub_map_;
  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr sub_trajectory_;
  rclcpp::Subscription<autoware_vehicle_msgs::msg::TurnIndicatorsCommand>::SharedPtr
    sub_turn_indicators_cmd_;
  rclcpp::Subscription<autoware_vehicle_msgs::msg::HazardLightsCommand>::SharedPtr
    sub_hazard_lights_cmd_;
  rclcpp::Subscription<autoware_vehicle_msgs::msg::GearCommand>::SharedPtr sub_gear_cmd_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_init_twist_;
  rclcpp::Subscription<autoware_vehicle_msgs::msg::Engage>::SharedPtr sub_engage_;

  // --- Services ---
  rclcpp::Service<autoware_vehicle_msgs::srv::ControlModeCommand>::SharedPtr srv_mode_req_;
  rclcpp::CallbackGroup::SharedPtr group_api_service_;
  tier4_api_utils::Service<tier4_external_api_msgs::srv::InitializePose>::SharedPtr srv_set_pose_;

  // --- TF ---
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // --- Timer ---
  rclcpp::TimerBase::SharedPtr on_timer_;

  // --- State Variables ---
  nav_msgs::msg::Odometry current_odometry_{};
  geometry_msgs::msg::AccelWithCovarianceStamped current_acceleration_{};
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr current_trajectory_ptr_{};
  std::deque<autoware_planning_msgs::msg::Trajectory::ConstSharedPtr> trajectory_history_{};
  autoware_vehicle_msgs::msg::TurnIndicatorsCommand::ConstSharedPtr
    current_turn_indicators_cmd_ptr_{};
  autoware_vehicle_msgs::msg::HazardLightsCommand::ConstSharedPtr current_hazard_lights_cmd_ptr_{};
  autoware_vehicle_msgs::msg::GearCommand current_gear_cmd_{};
  autoware_vehicle_msgs::msg::ControlModeReport current_control_mode_{};

  // [SPS MIMIC] Map Storage
  // We keep the pointer because it contains the Fast Search Index (R-Tree)
  lanelet::LaneletMapConstPtr lanelet_map_ptr_;
  bool has_map_ = false;

  bool is_initialized_ = false;
  // Whether the vehicle state may advance. Set by the engage command (/autoware/engage) and the
  // initial_engage_state parameter. While false, the vehicle is held in place until engaged.
  bool simulate_motion_ = false;
  std::string simulated_frame_id_ = "base_link";
  std::string origin_frame_id_ = "odom";

  double current_target_steering_ = 0.0;
  int n_delay_steps_{0};

  // Ego-advance algorithm selector: "snap" (follow the model prediction by plan index) or
  // "physics" (conventional SPS-mimic: nearest point + velocity Euler integration).
  std::string tracker_mode_ = "snap";
  // Re-planning cadence: adopt a fresh plan only every `replan_interval` steps (Python
  // replan_interval). 1 == adopt the latest plan whenever it updates.
  int replan_interval_{1};
  // Plan currently being followed, and the index cursor / step counter for the snap mode.
  autoware_planning_msgs::msg::Trajectory::ConstSharedPtr cached_plan_{};
  size_t plan_cursor_ = 0;
  size_t step_counter_ = 0;

  size_t last_closest_idx_ = 0;

  // --- Callbacks ---
  void on_map(const autoware_map_msgs::msg::LaneletMapBin::ConstSharedPtr msg);
  void on_trajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);
  void on_turn_indicators_cmd(
    const autoware_vehicle_msgs::msg::TurnIndicatorsCommand::ConstSharedPtr msg);
  void on_hazard_lights_cmd(
    const autoware_vehicle_msgs::msg::HazardLightsCommand::ConstSharedPtr msg);
  void on_gear_cmd(const autoware_vehicle_msgs::msg::GearCommand::ConstSharedPtr msg);
  void on_initialpose(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg);
  void on_initialtwist(const geometry_msgs::msg::TwistStamped::ConstSharedPtr msg);
  void on_engage(const autoware_vehicle_msgs::msg::Engage::ConstSharedPtr msg);

  void on_control_mode_request(
    const autoware_vehicle_msgs::srv::ControlModeCommand::Request::ConstSharedPtr request,
    const autoware_vehicle_msgs::srv::ControlModeCommand::Response::SharedPtr response);

  void on_set_pose(
    const tier4_external_api_msgs::srv::InitializePose::Request::ConstSharedPtr request,
    const tier4_external_api_msgs::srv::InitializePose::Response::SharedPtr response);

  void on_timer();

  // --- Logic Helpers ---
  void set_initial_state(
    const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Twist & twist);

  // [SPS MIMIC] Conventional advance: nearest trajectory point + velocity Euler integration.
  void update_state_from_trajectory_and_map(const double ego_pitch);
  // Snap advance: follow the model prediction by plan index (replan-hold + cursor).
  void update_state_snap();
  double calculate_ego_pitch() const;
  double get_z_pose_from_trajectory(
    const double x, const double y, const nav_msgs::msg::Odometry & prev_odometry);

  geometry_msgs::msg::TransformStamped get_transform_msg(
    const std::string parent_frame, const std::string child_frame);

  void publish_velocity(const autoware_vehicle_msgs::msg::VelocityReport & velocity);
  void publish_odometry(const nav_msgs::msg::Odometry & odometry);
  void publish_steering(const autoware_vehicle_msgs::msg::SteeringReport & steer);
  void publish_acceleration();
  void publish_imu();
  void publish_tf(const nav_msgs::msg::Odometry & odometry);
};

}  // namespace autoware::simulator::perfect_tracker

#endif  // AUTOWARE__PERFECT_TRACKER__PERFECT_TRACKER_NODE_HPP_
