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

"""Start the installed MPPI Streamlit application."""

from pathlib import Path
import sys

from streamlit.web import cli


def main() -> int:
    application = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(application), *sys.argv[1:]]
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
