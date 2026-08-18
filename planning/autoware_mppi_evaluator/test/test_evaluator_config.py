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

from pathlib import Path
import tempfile
import unittest

from autoware_mppi_evaluator.evaluator_config import make_configuration


class CostParams:
    def __init__(self):
        self.lambda_ = 1500.0
        self.track_terminal_scale = 10.0
        self.remaining_distance_coeff = 0.0
        self.path_overshoot_coeff = 0.0
        self.track_center_coeff = 0.0
        self.accel_cmd_std_dev = 0.35
        self.steer_cmd_std_dev = 0.024
        self.nominal_curvature_min_chord_length_m = 1.5
        self.lateral_boundary_soft_margin = 0.2
        self.obstacle_safe_margin = 0.5
        self.road_border_safe_margin = 0.3
        self.drivable_area_safe_margin = 0.0
        self.drivable_area_barrier_weight = 2000.0
        self.crash_contact_penalty = 100000.0


class RuntimeOptions:
    use_last_control_as_nominal = False
    use_temporal_mpt_as_nominal = False
    enable_input_delay_compensation = True


class VehicleParams:
    pass


class Configuration:
    pass


class Backend:
    CostParams = CostParams
    RuntimeOptions = RuntimeOptions
    VehicleParams = VehicleParams
    Configuration = Configuration


def write_parameters(directory: str, contents: str) -> str:
    path = Path(directory) / "parameters.yaml"
    path.write_text("/**:\n  ros__parameters:\n" + contents, encoding="utf-8")
    return str(path)


class TestEvaluatorConfig(unittest.TestCase):
    def test_loads_current_cost_and_runtime_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_parameters(
                directory,
                "    lambda: 500.0\n"
                "    track_terminal_scale: 12.0\n"
                "    remaining_distance_coeff: 150.0\n"
                "    path_overshoot_coeff: 25.0\n"
                "    track_center_coeff: 20.0\n"
                "    accel_cmd_std_dev: 0.1\n"
                "    steer_cmd_std_dev: 0.02\n"
                "    nominal_curvature_min_chord_length_m: 2.0\n"
                "    lateral_boundary_soft_margin: 0.25\n"
                "    obstacle_safe_margin: 0.6\n"
                "    road_border_safe_margin: 0.4\n"
                "    drivable_area_safe_margin: 0.1\n"
                "    drivable_area_barrier_weight: 2500.0\n"
                "    crash_contact_penalty: 120000.0\n"
                "    use_last_control_as_nominal: true\n"
                "    use_temporal_mpt_as_nominal: true\n"
                "    enable_input_delay_compensation: false\n",
            )

            configuration = make_configuration(Backend, optimizer_path=path)

            self.assertEqual(configuration.cost_params.lambda_, 500.0)
            self.assertEqual(configuration.cost_params.track_terminal_scale, 12.0)
            self.assertEqual(configuration.cost_params.remaining_distance_coeff, 150.0)
            self.assertEqual(configuration.cost_params.path_overshoot_coeff, 25.0)
            self.assertEqual(configuration.cost_params.track_center_coeff, 20.0)
            self.assertEqual(configuration.cost_params.accel_cmd_std_dev, 0.1)
            self.assertEqual(configuration.cost_params.steer_cmd_std_dev, 0.02)
            self.assertEqual(configuration.cost_params.nominal_curvature_min_chord_length_m, 2.0)
            self.assertEqual(configuration.cost_params.lateral_boundary_soft_margin, 0.25)
            self.assertEqual(configuration.cost_params.obstacle_safe_margin, 0.6)
            self.assertEqual(configuration.cost_params.road_border_safe_margin, 0.4)
            self.assertEqual(configuration.cost_params.drivable_area_safe_margin, 0.1)
            self.assertEqual(configuration.cost_params.drivable_area_barrier_weight, 2500.0)
            self.assertEqual(configuration.cost_params.crash_contact_penalty, 120000.0)
            self.assertTrue(configuration.runtime_options.use_last_control_as_nominal)
            self.assertTrue(configuration.runtime_options.use_temporal_mpt_as_nominal)
            self.assertFalse(configuration.runtime_options.enable_input_delay_compensation)

    def test_rejects_an_optimizer_parameter_missing_from_the_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_parameters(directory, "    newly_added_parameter: 1.0\n")

            with self.assertRaisesRegex(ValueError, "newly_added_parameter"):
                make_configuration(Backend, optimizer_path=path)


if __name__ == "__main__":
    unittest.main()
