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
