#!/usr/bin/env python3
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
