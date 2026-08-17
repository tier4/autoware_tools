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


class TerminalCostParams:
    def __init__(self):
        self.track = 10000.0
        self.heading = 5000.0
        self.lateral_distance = 0.0
        self.lateral_yaw_error = 0.0
        self.track_center = 0.0


class CostParams:
    def __init__(self):
        self.lambda_ = 1500.0
        self.track_center_coeff = 0.0
        self.nominal_spline_smoothing_weight = 10.0
        self.obstacle_safe_margin = 0.5
        self.terminal_coeffs = TerminalCostParams()


class RuntimeOptions:
    curvature_std = 0.00625
    min_optimization_length = 0.0


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
                "    track_center_coeff: 20.0\n"
                "    nominal_spline_smoothing_weight: 25.0\n"
                "    obstacle_safe_margin: 0.7\n"
                "    terminal_coeffs:\n"
                "      track: 1234.0\n"
                "      lateral_distance: 42.0\n"
                "    curvature_std: 0.01\n"
                "    min_optimization_length: 3.5\n",
            )

            configuration = make_configuration(Backend, optimizer_path=path)

            self.assertEqual(configuration.cost_params.lambda_, 500.0)
            self.assertEqual(configuration.cost_params.track_center_coeff, 20.0)
            self.assertEqual(configuration.cost_params.nominal_spline_smoothing_weight, 25.0)
            self.assertEqual(configuration.cost_params.obstacle_safe_margin, 0.7)
            self.assertEqual(configuration.cost_params.terminal_coeffs.track, 1234.0)
            self.assertEqual(configuration.cost_params.terminal_coeffs.heading, 5000.0)
            self.assertEqual(configuration.cost_params.terminal_coeffs.lateral_distance, 42.0)
            self.assertEqual(configuration.runtime_options.curvature_std, 0.01)
            self.assertEqual(configuration.runtime_options.min_optimization_length, 3.5)

    def test_rejects_an_optimizer_parameter_missing_from_the_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_parameters(directory, "    newly_added_parameter: 1.0\n")

            with self.assertRaisesRegex(ValueError, "newly_added_parameter"):
                make_configuration(Backend, optimizer_path=path)

    def test_rejects_an_unknown_nested_terminal_coefficient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_parameters(
                directory,
                "    terminal_coeffs:\n" "      newly_added_coefficient: 1.0\n",
            )

            with self.assertRaisesRegex(ValueError, r"terminal_coeffs\.newly_added_coefficient"):
                make_configuration(Backend, optimizer_path=path)


if __name__ == "__main__":
    unittest.main()
