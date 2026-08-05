//
//  Copyright 2024 Tier IV, Inc. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#ifndef CREEP_GUIDANCE_PANEL_HPP_
#define CREEP_GUIDANCE_PANEL_HPP_

#include <QLabel>
#include <QPushButton>
#include <QString>
#include <QTableWidget>

#ifndef Q_MOC_RUN
// cpp
#include <string>
// ros
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
// autoware
#include <tier4_creep_guidance_msgs/msg/creep_status.hpp>
#include <tier4_creep_guidance_msgs/msg/creep_status_array.hpp>
#include <tier4_creep_guidance_msgs/srv/creep_trigger_command.hpp>
#endif

namespace rviz_plugins
{
using tier4_creep_guidance_msgs::msg::Command;
using tier4_creep_guidance_msgs::msg::CreepStatus;
using tier4_creep_guidance_msgs::msg::CreepStatusArray;
using tier4_creep_guidance_msgs::msg::Module;
using tier4_creep_guidance_msgs::msg::State;
using tier4_creep_guidance_msgs::srv::CreepTriggerCommand;

class CreepGuidancePanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit CreepGuidancePanel(QWidget * parent = nullptr);
  void onInitialize() override;

protected:
  void on_creep_status(const CreepStatusArray::ConstSharedPtr msg);

private:
  void on_button_clicked();

  static std::string get_module_name(const Module & module);
  static std::string get_status_name(const State & state);
  static std::string get_command_name(const Command & command);

  rclcpp::Node::SharedPtr raw_node_;
  rclcpp::Subscription<CreepStatusArray>::SharedPtr sub_creep_status_;
  rclcpp::Client<CreepTriggerCommand>::SharedPtr cli_creep_trigger_;

  CreepStatusArray::ConstSharedPtr current_status_;

  QLabel * status_label_;
  QTableWidget * status_table_;
  QPushButton * action_button_;
};

}  // namespace rviz_plugins

#endif  // CREEP_GUIDANCE_PANEL_HPP_
