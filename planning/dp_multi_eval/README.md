# dp_multi_eval

Parallel-friendly **Diffusion Planner evaluation** for Autoware planning simulator.

Two ways to run the same pipeline:

| Mode | Description | Documentation |
|------|-------------|---------------|
| **Headless** | CLI batch — no RViz, no Streamlit; `run_all.py` + HTML dashboard | **[docs/HEADLESS.md](docs/HEADLESS.md)** |
| **GUI** | Streamlit wrapper around the same orchestrator | [docs/GUI_TESTING.md](docs/GUI_TESTING.md) |

**Status:** early integration — see [docs/REVIEW.md](docs/REVIEW.md) for code review.

---

## Documentation index

| Doc | For |
|-----|-----|
| **[docs/HEADLESS.md](docs/HEADLESS.md)** | **Headless batch pipeline** — config, CLI, resume, metrics, engineer test checklist |
| [docs/GUI_TESTING.md](docs/GUI_TESTING.md) | Streamlit `app.py` test checklist |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Shared ops (metrics detail, troubleshooting) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design, phases, code map |
| [docs/REVIEW.md](docs/REVIEW.md) | PR review checklist |

Related: [diffusion_planner_batch_eval](../diffusion_planner_batch_eval/README.md) — interactive RViz-based debugging.

---

## 30-second start (headless)

```bash
colcon build --packages-select dp_multi_eval diffusion_planner_batch_eval --allow-overriding dp_multi_eval
source install/setup.bash
cp src/tools/planning/dp_multi_eval/config/example_pipeline.yaml ~/dp_multi_eval_pipeline.yaml
# edit paths

ros2 run dp_multi_eval run_all.py -c ~/dp_multi_eval_pipeline.yaml --max_workers 1
```

Full guide: **[docs/HEADLESS.md](docs/HEADLESS.md)**

---

## Maintainer

Takahiko Hasegawa — see `package.xml`.
