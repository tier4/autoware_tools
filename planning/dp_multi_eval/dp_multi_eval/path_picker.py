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

"""Native folder picker for the Streamlit GUI (tkinter, desktop only)."""

from __future__ import annotations

from pathlib import Path


def pick_folder(title: str, initial: str | Path = "") -> str | None:
    """Open an OS folder dialog. Returns None if cancelled or unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    initial_dir = str(Path(initial).expanduser()) if initial else str(Path.home())
    if not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    root = tk.Tk()
    root.withdraw()
    try:
        root.wm_attributes("-topmost", 1)
    except tk.TclError:
        pass
    try:
        selected = filedialog.askdirectory(title=title, initialdir=initial_dir)
    finally:
        root.destroy()
    return selected or None
