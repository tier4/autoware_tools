#!/usr/bin/env python3

# Copyright 2026 TIER IV, Inc.
# Licensed under the Apache License, Version 2.0

"""CLI entry for the scene driver: the original perception_reproducer on dp_multi_eval's clock."""

from __future__ import annotations

import sys


def main() -> int:
    from dp_multi_eval.scene_driver import main as driver_main

    return driver_main()


if __name__ == "__main__":
    sys.exit(main())
