#!/usr/bin/env python3
"""Run every test script of dp_multi_eval.

    python3 tests/run_tests.py            # summary table, exit 1 on any FAIL
    python3 tests/run_tests.py -v         # also print the output of failures
    python3 tests/run_tests.py -k light   # only tests whose name contains "light"

Each test is a plain script: exit 0 = PASS; exit 0 with a last line starting
with "SKIP" = SKIP (a prerequisite such as a sourced workspace or the map is
missing); anything else = FAIL. Integration tests skip themselves; run with
the Autoware workspace sourced to exercise them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = [HERE]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-k", default="", help="run only tests whose file name contains this")
    args = p.parse_args()

    results = []
    for suite in SUITES:
        if not suite.is_dir():
            continue
        for test in sorted(suite.glob("test_*.py")):
            if args.k and args.k not in test.name:
                continue
            t0 = time.monotonic()
            proc = subprocess.run(
                [sys.executable, str(test)], cwd=str(suite.parent), capture_output=True, text=True, timeout=900
            )
            out = (proc.stdout + proc.stderr).strip()
            last = out.splitlines()[-1] if out else ""
            if proc.returncode == 0 and last.startswith("SKIP"):
                status = "SKIP"
            elif proc.returncode == 0:
                status = "PASS"
            else:
                status = "FAIL"
            results.append((status, suite.parent.name, test.name, time.monotonic() - t0, last, out))

    width = max(len(r[2]) for r in results) if results else 20
    for status, pkg, name, dt, last, _ in results:
        print(f"{status:4} {pkg}/{name:<{width}} {dt:5.1f}s  {last[:90]}")
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "SKIP", "FAIL")}
    print(f"\n{counts['PASS']} passed, {counts['SKIP']} skipped, {counts['FAIL']} failed")
    if args.verbose:
        for status, pkg, name, _, _, out in results:
            if status == "FAIL":
                print(f"\n===== {pkg}/{name}\n{out[-4000:]}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
