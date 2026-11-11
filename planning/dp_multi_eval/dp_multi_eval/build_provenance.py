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

"""What the planner build and the checkpoint each expect, recorded with results.

The evaluation depends on two version numbers agreeing: the model's
``major_version`` and the planner's ``WEIGHT_MAJOR_VERSION``. When they do not,
the planner loads nothing and every run fails the same way. When they do, the
results are only as trustworthy as the reason they agree.

That reason is not always visible. The Autoware tree here is not under version
control -- ``src/`` is gitignored and there is no repository beneath it -- so a
local edit to constants.hpp leaves no trace in any status output, and a run can
depend on a patched build with nothing in the results to say so. Recording both
numbers next to the metrics makes the dependency part of the result instead of
part of somebody's memory.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

_CONSTANTS_REL = (
    "universe/planning/autoware_diffusion_planner/include/autoware/"
    "diffusion_planner/constants.hpp"
)
# WEIGHT_MAJOR_VERSION: the checkpoint must match exactly (older planners).
# WEIGHT_MINIMUM_VERSION: it must be at least this (since the 2026-10 planner).
_PATTERN = re.compile(r"WEIGHT_(MAJOR|MINIMUM)_VERSION\s*=\s*(\d+)")


def planner_weight_rule(autoware_root: Path | None = None) -> tuple[str, int] | None:
    """("exact" | "minimum", version) the planner build checks checkpoints
    against, if readable."""
    if autoware_root is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "src" / "autoware"
            if candidate.is_dir():
                autoware_root = candidate
                break
    if autoware_root is None:
        return None
    header = Path(autoware_root) / _CONSTANTS_REL
    try:
        match = _PATTERN.search(header.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    if not match:
        return None
    return ("exact" if match.group(1) == "MAJOR" else "minimum"), int(match.group(2))


def planner_weight_version(autoware_root: Path | None = None) -> int | None:
    """The version number of the planner's rule (exact or minimum)."""
    rule = planner_weight_rule(autoware_root)
    return rule[1] if rule else None


def checkpoint_version(model_config: Path | str | None) -> int | None:
    """major_version declared by the checkpoint: the args.json of a model
    directory (what the pipeline config names), or the one beside the ONNX a
    param YAML points at. A directory read as a YAML returned None, so every
    result carried no versions at all (2026-11-05 review)."""
    if not model_config:
        return None
    path = Path(model_config).expanduser()
    if path.is_dir():
        args = path / "args.json"
    else:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        onnx = re.search(r"onnx_model_path:\s*(\S+)", raw)
        if not onnx:
            return None
        args = Path(onnx.group(1)).parent / "args.json"
    try:
        value = json.loads(args.read_text(encoding="utf-8")).get("major_version")
    except (OSError, json.JSONDecodeError):
        return None
    return int(value) if isinstance(value, int) else None


# This file sits at <ws>/src/tools/planning/dp_multi_eval/dp_multi_eval/.
_WS_SRC = Path(__file__).resolve().parents[4]
_REPOS = {
    "dp_multi_eval": Path(__file__).resolve().parent,
    "universe": _WS_SRC / "autoware" / "universe",
    "launcher": _WS_SRC / "autoware" / "launcher",
}


def repo_state(path: Path) -> dict[str, Any] | None:
    """Commit, branch and whether tracked files differ from it; None off git."""
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL, timeout=30
        ).strip()
    try:
        return {
            "commit": git("rev-parse", "--short=10", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),   # "HEAD" when detached
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (subprocess.SubprocessError, OSError):
        return None


def build_provenance(model_config: Path | str | None) -> dict[str, Any]:
    """Both versions and whether they agree, plus the code each result came
    from: on 2026-11-05 eight H81 runs from as many tool versions sat side by
    side in test_res with nothing to tell them apart."""
    rule = planner_weight_rule()
    checkpoint = checkpoint_version(model_config)
    out: dict[str, Any] = {
        "planner_weight_major_version": rule[1] if rule else None,
        "planner_weight_rule": rule[0] if rule else None,
        "checkpoint_major_version": checkpoint,
        "repos": {name: repo_state(path) for name, path in _REPOS.items() if path.is_dir()},
    }
    if rule is not None and checkpoint is not None:
        kind, version = rule
        out["versions_agree"] = checkpoint == version if kind == "exact" else checkpoint >= version
    return out
