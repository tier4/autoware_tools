# Legacy

Kept in the tree, not used for current work, not maintained against new features.

- **Streamlit GUI** (`ros2 run dp_multi_eval app.py`, needs `streamlit`): checkbox
  selection of bags/models, start/stop, embedded dashboard. Writes
  `results_root/<run>/pipeline_config.yaml` with the selection as `bag_whitelist`. Knows
  nothing of `rescore.py` or `_archive/`. The CLI with `--only` does the same.
- **Scenario Simulator mode** (`mode: scenario_simulator`): runs `.yaml`/`.xosc`
  scenarios via `scenario_test_runner`. No recording → no legs or
  recording-based rules. Unused since 2026-09-10.
- **Continuous drive loop** (`window_eval: false`): the pre-leg harness with stall
  detection, nudges, repositioning and re-sync. Its knobs remain in the yaml and
  its whole-bag metrics gate a run that has no `windows.json`.
- **`diffusion_planner_batch_eval`**: the RViz-based predecessor; only its
  `route_setup.py` is still imported.
