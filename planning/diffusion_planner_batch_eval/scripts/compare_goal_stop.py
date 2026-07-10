#!/usr/bin/env python3

# Copyright 2025 TIER IV, Inc.
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

"""Compare goal-stop accuracy across multiple models.

Reads per_model_per_bag_goal_stop.csv (from analyze_model_comparison.py) and prints a
per-model summary plus an optional scatter / bar plot of stop offsets in the goal frame.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRIC_COLUMNS = (
    "goal_stop_position_m",
    "goal_stop_lateral_m",
    "goal_stop_longitudinal_m",
    "goal_stop_heading_deg",
)


def load_goal_stop_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def filter_rows(
    rows: list[dict[str, str]],
    models: list[str] | None,
    success_only: bool,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if row.get("goal_stop_status") != "ok":
            continue
        if success_only and row.get("run_status") != "success":
            continue
        if models and row.get("model") not in models:
            continue
        out.append(row)
    return out


def print_status_breakdown(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("[warn] CSV has no rows.")
        return
    gs_counts: dict[str, int] = defaultdict(int)
    run_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        gs_counts[row.get("goal_stop_status") or "(empty)"] += 1
        run_counts[row.get("run_status") or "(empty)"] += 1
    print("\nRow breakdown in CSV:")
    print("  goal_stop_status:")
    for key in sorted(gs_counts):
        print(f"    {key}: {gs_counts[key]}")
    print("  run_status:")
    for key in sorted(run_counts):
        print(f"    {key}: {run_counts[key]}")
    ok_success = sum(
        1
        for row in rows
        if row.get("goal_stop_status") == "ok" and row.get("run_status") == "success"
    )
    ok_any = sum(1 for row in rows if row.get("goal_stop_status") == "ok")
    print(f"  usable (goal_stop_status=ok): {ok_any}/{len(rows)}")
    print(f"  usable (ok + run_status=success): {ok_success}/{len(rows)}")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _finite_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = _to_float(row.get(column, ""))
        if not math.isnan(value):
            values.append(value)
    return values


def summarize(rows: list[dict[str, str]], models: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model in models:
        model_rows = [row for row in rows if row.get("model") == model]
        if not model_rows:
            continue

        pos = _finite_values(model_rows, "goal_stop_position_m")
        lat_signed = _finite_values(model_rows, "goal_stop_lateral_m")
        lat_abs = [abs(v) for v in lat_signed]
        lon_signed = _finite_values(model_rows, "goal_stop_longitudinal_m")
        head_signed = _finite_values(model_rows, "goal_stop_heading_deg")
        head_abs = [abs(v) for v in head_signed]
        speed = _finite_values(model_rows, "goal_stop_speed_mps")
        stop_time = _finite_values(model_rows, "goal_stop_time_sec")

        if not pos:
            continue

        summary[model] = {
            "bags": float(len(pos)),
            "mean_position_m": statistics.fmean(pos),
            "median_position_m": statistics.median(pos),
            "p95_position_m": _percentile(pos, 95.0),
            "mean_lateral_m": statistics.fmean(lat_signed) if lat_signed else float("nan"),
            "mean_abs_lateral_m": statistics.fmean(lat_abs) if lat_abs else float("nan"),
            "mean_longitudinal_m": statistics.fmean(lon_signed) if lon_signed else float("nan"),
            "mean_heading_deg": statistics.fmean(head_signed) if head_signed else float("nan"),
            "mean_abs_heading_deg": statistics.fmean(head_abs) if head_abs else float("nan"),
            "mean_stop_speed_mps": statistics.fmean(speed) if speed else float("nan"),
            "mean_stop_time_sec": statistics.fmean(stop_time) if stop_time else float("nan"),
        }
    return summary


def _short_name(model: str, max_len: int = 48) -> str:
    return model if len(model) <= max_len else model[: max_len - 3] + "..."


def print_summary(summary: dict[str, dict[str, float]]) -> None:
    if not summary:
        print("[warn] No goal-stop rows to summarize (check status / model filters).")
        return

    print("\nGoal-stop accuracy (lower position / |lat| / |head| is better)")
    print("  Signed lateral/longitudinal/heading = mean bias (+ left/ahead, - right/behind)")
    print("  |lat| / |head| = mean magnitude (errors do not cancel left vs right)")

    acc_header = (
        f"  {'model':<48} {'bags':>5} {'position':>9} {'median':>8} "
        f"{'p95':>8} {'|lat|':>7} {'|head|':>7}"
    )
    print(f"\n{acc_header}")
    print("  " + "-" * (len(acc_header) - 2))
    for model in sorted(summary, key=lambda m: summary[m]["mean_position_m"]):
        s = summary[model]
        print(
            f"  {_short_name(model):<48} {int(s['bags']):>5} "
            f"{s['mean_position_m']:>9.2f} {s['median_position_m']:>8.2f} "
            f"{s['p95_position_m']:>8.2f} {s['mean_abs_lateral_m']:>7.2f} "
            f"{s['mean_abs_heading_deg']:>7.2f}"
        )

    bias_header = (
        f"  {'model':<48} {'lat_bias':>9} {'long_bias':>10} "
        f"{'head_bias':>10} {'speed':>7} {'time':>7}"
    )
    print(f"\n{bias_header}")
    print("  " + "-" * (len(bias_header) - 2))
    for model in sorted(summary, key=lambda m: summary[m]["mean_position_m"]):
        s = summary[model]
        print(
            f"  {_short_name(model):<48} "
            f"{s['mean_lateral_m']:>9.2f} {s['mean_longitudinal_m']:>10.2f} "
            f"{s['mean_heading_deg']:>10.2f} {s['mean_stop_speed_mps']:>7.2f} "
            f"{s['mean_stop_time_sec']:>7.1f}"
        )


def shorten(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def plot_comparison(
    rows: list[dict[str, str]],
    models: list[str],
    summary: dict[str, dict[str, float]],
    save_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(models):
        color = colors[idx % len(colors)]
        xs = []
        ys = []
        for row in rows:
            if row.get("model") != model:
                continue
            lon = _to_float(row["goal_stop_longitudinal_m"])
            lat = _to_float(row["goal_stop_lateral_m"])
            if math.isnan(lon) or math.isnan(lat):
                continue
            xs.append(lon)
            ys.append(lat)
        if xs:
            ax_scatter.scatter(
                xs, ys, color=color, alpha=0.7, s=40, label=shorten(model), edgecolors="none"
            )

    ax_scatter.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_scatter.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_scatter.plot(0, 0, marker="*", color="red", markersize=18, label="goal", zorder=5)
    ax_scatter.set_xlabel("longitudinal offset [m]  (+ ahead / − behind goal)")
    ax_scatter.set_ylabel("lateral offset [m]  (+ left / − right)")
    ax_scatter.set_title("Stop pose vs goal (goal frame)")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.axis("equal")
    ax_scatter.legend(loc="best", fontsize=8)

    ordered = sorted(summary, key=lambda m: summary[m]["mean_position_m"])
    labels = [shorten(m) for m in ordered]
    means = [summary[m]["mean_position_m"] for m in ordered]
    p95s = [summary[m]["p95_position_m"] for m in ordered]
    bar_colors = [colors[models.index(m) % len(colors)] for m in ordered]
    x = range(len(ordered))
    ax_bar.bar(x, means, color=bar_colors, alpha=0.8, label="mean")
    ax_bar.plot(list(x), p95s, "k_", markersize=18, markeredgewidth=2, label="p95")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylabel("position error to goal [m]")
    ax_bar.set_title("Mean goal-stop distance (lower = better)")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[info] Saved {save_path}")
        plt.close(fig)
    else:
        print("\n[info] Interactive plot — close window to exit")
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare goal-stop accuracy across models from per_model_per_bag_goal_stop.csv."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="Batch eval output dir (reads quantitative_analysis/)"
    )
    parser.add_argument("--csv", type=Path, help="Direct path to per_model_per_bag_goal_stop.csv")
    parser.add_argument("--models", help="Comma-separated model names (default: all in CSV)")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only runs with run_status == success (default: all runs with goal_stop_status=ok)",
    )
    parser.add_argument("--plot", action="store_true", help="Show scatter + bar comparison plot")
    parser.add_argument("--save", type=Path, help="Save plot PNG (implies --plot)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv.expanduser()
    elif args.output_dir:
        csv_path = (
            args.output_dir.expanduser()
            / "quantitative_analysis"
            / "per_model_per_bag_goal_stop.csv"
        )
    else:
        print("[error] Pass -o OUTPUT_DIR or --csv PATH")
        sys.exit(1)

    try:
        all_rows = load_goal_stop_csv(csv_path)
    except FileNotFoundError as error:
        print(f"[error] {error}")
        print("Run analyze_model_comparison.py first to generate the goal-stop CSV.")
        sys.exit(1)

    models = (
        [name.strip() for name in args.models.split(",") if name.strip()]
        if args.models
        else sorted({row["model"] for row in all_rows if row.get("model")})
    )

    rows = filter_rows(all_rows, models, success_only=args.success_only)
    if not rows:
        print_status_breakdown(all_rows)
        if not args.success_only:
            print(
                "\n[hint] No rows with goal_stop_status=ok. Common causes:\n"
                "  - bag_path_unresolved: pass -c config.yaml with rosbag_dir, or ensure metadata.json\n"
                "    in each trace dir has bag_path\n"
                "  - Re-run: ros2 run diffusion_planner_batch_eval analyze_model_comparison.py "
                "-o $RESULTS -c ~/diffusion_batch_config.yaml"
            )
        else:
            print(
                "\n[hint] Try without --success-only to include stuck/timeout runs "
                "(stop pose is still computed from the trace)."
            )
        sys.exit(1)

    summary = summarize(rows, models)
    print_summary(summary)

    if args.success_only:
        print("\n[info] Filter: goal_stop_status=ok and run_status=success")
    else:
        print("\n[info] Filter: goal_stop_status=ok (all run statuses). Use --success-only for arrivals only.")

    if args.plot or args.save:
        if not summary:
            print("[warn] Nothing to plot.")
            return
        plot_comparison(rows, models, summary, args.save)


if __name__ == "__main__":
    main()
