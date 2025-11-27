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

#include "creep_guidance_panel.hpp"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QVBoxLayout>
#include <rviz_common/display_context.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rviz_plugins
{
using std::placeholders::_1;

CreepGuidancePanel::CreepGuidancePanel(QWidget * parent) : rviz_common::Panel(parent)
{
  auto * v_layout = new QVBoxLayout;

  // Status label
  status_label_ = new QLabel("Waiting for creep guidance status...");
  status_label_->setAlignment(Qt::AlignCenter);
  v_layout->addWidget(status_label_);

  // Status table
  auto vertical_header = new QHeaderView(Qt::Vertical);
  vertical_header->hide();
  auto horizontal_header = new QHeaderView(Qt::Horizontal);
  horizontal_header->setSectionResizeMode(QHeaderView::Stretch);

  status_table_ = new QTableWidget();
  status_table_->setColumnCount(6);
  status_table_->setHorizontalHeaderLabels(
    {"ID", "Module", "State", "Command", "Start Distance", "Finish Distance"});
  status_table_->setVerticalHeader(vertical_header);
  status_table_->setHorizontalHeader(horizontal_header);

  v_layout->addWidget(status_table_);

  // Action button
  action_button_ = new QPushButton("Trigger Creep");
  connect(action_button_, &QPushButton::clicked, this, &CreepGuidancePanel::on_button_clicked);
  v_layout->addWidget(action_button_);

  setLayout(v_layout);
}

std::string CreepGuidancePanel::get_module_name(const Module & module)
{
  switch (module.type) {
    case Module::NONE:
      return "None";
    case Module::CROSSWALK:
      return "Crosswalk";
    case Module::INTERSECTION:
      return "Intersection";
    case Module::INTERSECTION_OCCLUSION:
      return "Intersection Occlusion";
    default:
      return "Unknown";
  }
}

std::string CreepGuidancePanel::get_status_name(const State & state)
{
  switch (state.type) {
    case State::DEACTIVATED:
      return "Deactivated";
    case State::ACTIVATED:
      return "Activated";
    default:
      return "Unknown";
  }
}

std::string CreepGuidancePanel::get_command_name(const Command & command)
{
  switch (command.type) {
    case Command::DEACTIVATE:
      return "Deactivate";
    case Command::ACTIVATE:
      return "Activate";
    default:
      return "Unknown";
  }
}

void CreepGuidancePanel::onInitialize()
{
  raw_node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  sub_creep_status_ = raw_node_->create_subscription<CreepStatusArray>(
    "/api/external/get/creep_status", 1, std::bind(&CreepGuidancePanel::on_creep_status, this, _1));

  // Create service client for creep trigger
  cli_creep_trigger_ =
    raw_node_->create_client<CreepTriggerCommand>("/api/external/set/creep_trigger_commands");
}

void CreepGuidancePanel::on_creep_status(const CreepStatusArray::ConstSharedPtr msg)
{
  // Store current status
  current_status_ = msg;

  status_table_->clearContents();

  // Filter statuses: finish_distance >= 0 and start_distance >= 0
  std::vector<CreepStatus> filtered_statuses;
  for (const auto & status : msg->statuses) {
    if (status.finish_distance >= 0.0 && status.start_distance >= 0.0) {
      filtered_statuses.push_back(status);
    }
  }

  // Sort by start_distance in ascending order
  std::sort(
    filtered_statuses.begin(), filtered_statuses.end(),
    [](const CreepStatus & a, const CreepStatus & b) {
      return a.start_distance < b.start_distance;
    });

  status_label_->setText(
    QString::fromStdString("Creep Statuses: " + std::to_string(filtered_statuses.size())));

  if (filtered_statuses.empty()) {
    status_table_->update();
    return;
  }

  status_table_->setRowCount(static_cast<int>(filtered_statuses.size()));

  int cnt = 0;
  for (const auto & status : filtered_statuses) {
    // ID
    {
      auto label = new QLabel(QString::number(status.id));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 0, label);
    }

    // Module
    {
      auto label = new QLabel(QString::fromStdString(get_module_name(status.module)));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 1, label);
    }

    // State
    {
      auto label = new QLabel(QString::fromStdString(get_status_name(status.state)));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 2, label);
    }

    // Command
    {
      auto label = new QLabel(QString::fromStdString(get_command_name(status.command)));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 3, label);
    }

    // Start Distance
    {
      auto label = new QLabel(QString::number(status.start_distance, 'f', 2));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 4, label);
    }

    // Finish Distance
    {
      auto label = new QLabel(QString::number(status.finish_distance, 'f', 2));
      label->setAlignment(Qt::AlignCenter);
      status_table_->setCellWidget(cnt, 5, label);
    }

    cnt++;
  }

  status_table_->update();
}

void CreepGuidancePanel::on_button_clicked()
{
  // Check if current status is available
  if (!current_status_) {
    RCLCPP_WARN(raw_node_->get_logger(), "No current status available");
    return;
  }

  // Filter statuses with start_distance >= 0
  std::vector<CreepStatus> filtered_statuses;
  for (const auto & status : current_status_->statuses) {
    if (status.start_distance >= 0.0) {
      filtered_statuses.push_back(status);
    }
  }

  if (filtered_statuses.empty()) {
    RCLCPP_WARN(raw_node_->get_logger(), "No statuses with start_distance >= 0 found");
    return;
  }

  // Find the status with the minimum start_distance
  auto min_status = std::min_element(
    filtered_statuses.begin(), filtered_statuses.end(),
    [](const CreepStatus & a, const CreepStatus & b) {
      return a.start_distance < b.start_distance;
    });

  // Create service request
  auto request = std::make_shared<CreepTriggerCommand::Request>();
  request->stamp = raw_node_->now();

  // Add activate command only for the status with minimum start_distance
  tier4_creep_guidance_msgs::msg::CreepTriggerCommand command;
  command.id = min_status->id;
  command.module = min_status->module;
  command.command.type = tier4_creep_guidance_msgs::msg::Command::ACTIVATE;
  request->commands.push_back(command);

  RCLCPP_INFO(
    raw_node_->get_logger(),
    "Adding activate command for ID: %ld, Module: %s, Start Distance: %.2f", min_status->id,
    get_module_name(min_status->module).c_str(), min_status->start_distance);

  // Call service asynchronously
  if (!cli_creep_trigger_->service_is_ready()) {
    RCLCPP_WARN(raw_node_->get_logger(), "Service is not ready");
    return;
  }
  RCLCPP_INFO(raw_node_->get_logger(), "Service is ready");

  auto result = cli_creep_trigger_->async_send_request(request);
  auto future = result.future.share();

  // Handle response asynchronously
  auto callback =
    [this](const std::shared_future<CreepTriggerCommand::Response::SharedPtr> & future) {
      const auto & response = future.get();
      RCLCPP_INFO(
        raw_node_->get_logger(), "Received response with %zu results", response->responses.size());
      for (const auto & res : response->responses) {
        RCLCPP_INFO(
          raw_node_->get_logger(), "ID: %ld, Module: %d, Success: %d", res.id, res.module.type,
          res.success);
      }
    };

  // Set callback for when response is received
  std::thread([future = std::move(future), callback = std::move(callback)]() mutable {
    callback(future);
  }).detach();
}

}  // namespace rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_plugins::CreepGuidancePanel, rviz_common::Panel)
