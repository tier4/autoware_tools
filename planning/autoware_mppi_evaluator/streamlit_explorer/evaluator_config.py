# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load native evaluator parameters from Autoware ROS parameter files."""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import yaml


def _ros_parameters(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if "/**" in document:
        return document["/**"].get("ros__parameters", {})
    for value in document.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            return value["ros__parameters"]
    raise ValueError(f"No ros__parameters section exists in {path}")


def make_configuration(
    backend,
    optimizer_path: Optional[str] = None,
    vehicle_info_path: Optional[str] = None,
    simulator_model_path: Optional[str] = None,
    name: str = "default",
):
    configuration = backend.Configuration()
    configuration.name = name

    optimizer = _ros_parameters(optimizer_path)
    cost = backend.CostParams()
    runtime = backend.RuntimeOptions()
    unsupported = []
    for key, value in optimizer.items():
        if key == "terminal_coeffs":
            if not isinstance(value, dict) or not hasattr(cost, key):
                unsupported.append(key)
                continue
            terminal_coeffs = cost.terminal_coeffs
            for terminal_key, terminal_value in value.items():
                if hasattr(terminal_coeffs, terminal_key):
                    setattr(terminal_coeffs, terminal_key, terminal_value)
                else:
                    unsupported.append(f"{key}.{terminal_key}")
            cost.terminal_coeffs = terminal_coeffs
            continue
        target_key = "lambda_" if key == "lambda" else key
        recognized = False
        if hasattr(cost, target_key):
            setattr(cost, target_key, value)
            recognized = True
        if hasattr(runtime, key):
            setattr(runtime, key, value)
            recognized = True
        if not recognized:
            unsupported.append(key)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported MPPI optimizer parameters in {optimizer_path}: {names}")
    configuration.cost_params = cost
    configuration.runtime_options = runtime

    vehicle_info = _ros_parameters(vehicle_info_path)
    simulator = _ros_parameters(simulator_model_path)
    vehicle = backend.VehicleParams()
    if vehicle_info:
        wheel_base = float(vehicle_info["wheel_base"])
        front_overhang = float(vehicle_info["front_overhang"])
        rear_overhang = float(vehicle_info["rear_overhang"])
        wheel_tread = float(vehicle_info["wheel_tread"])
        left_overhang = float(vehicle_info["left_overhang"])
        right_overhang = float(vehicle_info["right_overhang"])
        vehicle.ego_length = front_overhang + wheel_base + rear_overhang
        vehicle.ego_width = wheel_tread + left_overhang + right_overhang
        vehicle.ego_axle_to_box_center = 0.5 * vehicle.ego_length - rear_overhang
        vehicle.wheel_base = wheel_base
        vehicle.max_steer_angle = float(vehicle_info["max_steer_angle"])

    for key in (
        "acc_time_constant",
        "steer_time_constant",
        "steer_rate_lim",
        "vel_rate_lim",
        "acc_time_delay",
        "steer_time_delay",
    ):
        if key in simulator:
            setattr(vehicle, key, simulator[key])
    configuration.vehicle_params = vehicle
    return configuration
