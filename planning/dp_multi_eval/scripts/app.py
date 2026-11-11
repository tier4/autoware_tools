#!/usr/bin/env python3

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

"""Launch Streamlit GUI: ros2 run dp_multi_eval app.py"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("[error] pip install streamlit", file=sys.stderr)
        return 1
    app = Path(__file__).resolve().parents[1] / "dp_multi_eval" / "app.py"
    # When installed, prefer the python module path
    try:
        import dp_multi_eval.app as app_mod

        app = Path(app_mod.__file__).resolve()
    except Exception:  # noqa: BLE001
        pass
    sys.argv = ["streamlit", "run", str(app), "--server.headless=true"]
    stcli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
