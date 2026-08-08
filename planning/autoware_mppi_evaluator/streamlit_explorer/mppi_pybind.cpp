// Copyright 2026 TIER IV, Inc.
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
#include "autoware/mppi_optimizer/first_order_dubins_mppi_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
using autoware::mppi_optimizer;

PYBIND11_MODULE(mppi_optimizer_py, m)
{
  py::class_<FirstOrderDubinsMppiCostParams>(m, "CostParams")
    .def(py::init<>())
    .def_readwrite(
      "obstacle_collision_margin", &FirstOrderDubinsMppiCostParams::obstacle_collision_margin)
    .def_readwrite("boundary_threshold", &FirstOrderDubinsMppiCostParams::boundary_threshold)
    .def_readwrite(
      "road_border_collision_margin",
      &FirstOrderDubinsMppiCostParams::road_border_collision_margin);

  py::class_<FirstOrderDubinsMppiRuntimeOptions>(m, "RuntimeOptions")
    .def(py::init<>())
    .def_readwrite("skip_if_invalid", &FirstOrderDubinsMppiRuntimeOptions::skip_if_invalid);

  py::class_<FirstOrderDubinsMppiInterface>(m, "MppiInterface")
    .def(py::init<>())
    .def("set_cost_params", &FirstOrderDubinsMppiInterface::setCostParams)
    .def("set_runtime_options", &FirstOrderDubinsMppiInterface::setRuntimeOptions)
    .def(
      "optimize", [](
                    FirstOrderDubinsMppiInterface & self, const std::string & traj_json,
                    const std::string & odom_json, const std::string & objects_json) {
        // Deserialize standard ROS 2 JSON into structs, invoke CUDA optimizer,
        // and return serialized JSON result
        // ...
        return "{\"status\": \"success\"}";
      });
}
