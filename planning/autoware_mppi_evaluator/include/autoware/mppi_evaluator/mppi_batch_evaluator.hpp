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

#ifndef AUTOWARE__MPPI_EVALUATOR__MPPI_BATCH_EVALUATOR_HPP_
#define AUTOWARE__MPPI_EVALUATOR__MPPI_BATCH_EVALUATOR_HPP_

#include "autoware/mppi_evaluator/mppi_evaluation_types.hpp"

#include <memory>
#include <vector>

namespace autoware::mppi_evaluator
{

class MppiEvaluationSession
{
public:
  MppiEvaluationSession(
    MppiEnvironment environment, MppiConfiguration configuration, EvaluationMode mode);
  ~MppiEvaluationSession();

  MppiEvaluationSession(const MppiEvaluationSession &) = delete;
  MppiEvaluationSession & operator=(const MppiEvaluationSession &) = delete;
  MppiEvaluationSession(MppiEvaluationSession &&) noexcept;
  MppiEvaluationSession & operator=(MppiEvaluationSession &&) noexcept;

  EvaluatedFrameResult evaluate(const MppiInputFrame & frame);
  void reset();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class MppiBatchEvaluator
{
public:
  explicit MppiBatchEvaluator(MppiEnvironment environment);

  std::vector<EvaluatedFrameResult> evaluate(
    const std::vector<MppiInputFrame> & frames,
    const std::vector<MppiConfiguration> & configurations, EvaluationMode mode) const;

private:
  MppiEnvironment environment_;
};

}  // namespace autoware::mppi_evaluator

#endif  // AUTOWARE__MPPI_EVALUATOR__MPPI_BATCH_EVALUATOR_HPP_
