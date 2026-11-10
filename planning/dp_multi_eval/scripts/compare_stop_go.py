"""Stop-go of every archived run of one bag, side by side with the recording.

    python3 scripts/compare_stop_go.py <bag_dir> <results_root> [bag_key]

Reads <results_root>/<model>/<bag_key> and <results_root>/_archive/<bag_key>__*.
"""
import sys, json, glob, os
sys.path.insert(0, "/home/kaan/projects/pilot-auto.x2.isuzu.erga2/src/tools/planning/dp_multi_eval")
from pathlib import Path
from dp_multi_eval.bag_reader import load_bag_series, load_reference_ego
from dp_multi_eval.stop_go import stop_go
BAG = Path(sys.argv[1]); ROOT = Path(sys.argv[2]); KEY = sys.argv[3] if len(sys.argv) > 3 else BAG.name
ref = load_reference_ego(BAG)
dirs = sorted(glob.glob(str(ROOT / "_archive" / f"{KEY}__*"))) + sorted(glob.glob(str(ROOT / "*" / KEY)))
print(f"{'run':<24s} {'stops':>5s} {'stop_s':>6s} {'creep_s':>7s} {'cycles':>6s} {'drive_s':>7s}  flag  where")
for d in dirs:
    d = Path(d)
    if not (d / "output" / "metadata.yaml").exists(): continue
    try:
        s = load_bag_series(d / "output", objects_topic=None, trajectory_topic=None, turn_indicators_status_topic=None, turn_indicators_cmd_topic=None)
    except Exception as exc:
        print(f"{d.name[-26:]:<28s} unreadable ({str(exc)[:40]})"); continue
    goals = []
    try:
        goals = [(g["x"], g["y"]) for g in json.load(open(d / "route_goals.json")).get("goals", [])]
    except Exception: pass
    m = stop_go(s.ego, ref, goals); r = m["run"]
    name = d.name.split("__")[-1] if "__" in d.name else "CURRENT"
    base = s.ego[0].stamp_sec
    def sect(t):
        e = min(s.ego, key=lambda x: abs(x.stamp_sec - base - t))
        return f"t{int(t)}"
    secs = {}
    secs = [t for t, _ in r["stops_at"]]
    print(f"{name:<24s} {r['unplanned_stops']:>5d} {r['unplanned_stop_sec']:>6.1f} {r['creep_sec']:>7.1f} {r['start_brake_cycles']:>6d} {s.ego[-1].stamp_sec - base:>7.0f}  {'FAIL' if m['flagged'] else 'ok  '}  {secs}")
rec = m["recording"]; print(f"{'RECORDING':<28s} {rec['unplanned_stops']:>5d} {rec['unplanned_stop_sec']:>6.1f} {rec['creep_sec']:>7.1f} {rec['start_brake_cycles']:>6d} {ref[-1].stamp_sec - ref[0].stamp_sec:>7.0f}")
