#!/usr/bin/env python3

# Copyright 2026 TIER IV, Inc.
# Licensed under the Apache License, Version 2.0

"""CLI entry for Phase 1 single-job runner."""

from __future__ import annotations

import sys


def main() -> int:
    from dp_multi_eval.run_single_job import main as run_main

    return run_main()


if __name__ == "__main__":
    sys.exit(main())
