"""The knobs one evaluation job runs with.

Its own module because both the drive loop and the route/engage helpers need it,
and importing it from either of those would make them import each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dp_multi_eval.topics import TopicSet


def ros_call_env(env: dict[str, str]) -> dict[str, str]:
    """Env for a one-shot ros2 CLI call.

    The daemon caches the graph, so a call made through it can answer from a stale
    view — a node that has just died still listed, a service not yet seen missing.
    These calls gate on exactly that, so they ask the graph directly.
    """
    return {**env, "ROS2_DISABLE_DAEMON": "1"}


@dataclass
class JobConfig:
    model_config: Path
    output_dir: Path
    domain_id: int
    bag_path: Path | None = None
    scenario_path: Path | None = None
    mode: str = "reproducer"  # reproducer | scenario_simulator
    map_path: Path = Path("/opt/autoware/maps")
    vehicle_model: str = "lv828l"
    sensor_model: str = "aip_x2_gen2"
    vehicle_id: str | None = None
    topics: TopicSet = field(default_factory=TopicSet)
    end_condition: str = "route_arrived"  # route_arrived | bag_duration
    bag_duration_margin_sec: float = 30.0
    route_timeout_sec: float = 900.0
    psim_startup_sec: float = 120.0
    route_setup_timeout_sec: float = 90.0
    post_arrival_sec: float = 3.0
    auto_engage_timeout_sec: float = 60.0
    perception_warmup_sec: float = 5.0
    perception_ready_timeout_sec: float = 45.0
    perception_ready_min_objects: int = 1
    perception_ready_stable_sec: float = 2.0
    # Fail the job instead of engaging Auto with empty objects (common when the
    # full input bag is still loading into perception_reproducer).
    require_perception_ready: bool = True
    # Extra wall-time budget for large bags: timeout =
    # max(perception_ready_timeout_sec, bag_load_base_sec + bag_duration * bag_load_duration_factor).
    bag_load_base_sec: float = 120.0
    bag_load_duration_factor: float = 1.0
    bag_load_timeout_cap_sec: float = 3600.0
    # Free GPU memory a job waits for before launching its planner. Each worker
    # builds its own TensorRT engine; two starting at the same instant on one card
    # give the second "Cuda Runtime (out of memory)", after which the planner stays
    # up and silent and the run reads as a stall that has nothing to do with the
    # planner. 0 disables the gate.
    gpu_min_free_mib: float = 2500.0
    gpu_wait_timeout_sec: float = 600.0
    # Speed handed to the ego when a stall recovery repositions it. Zero restores
    # the old behaviour of setting it down at rest, which cannot break a stall
    # that StopPointFixer is holding — that plugin only acts on a stopped ego, so
    # an ego left at rest re-arms it immediately. Just above its 0.25 m/s
    # threshold is enough; the model does the rest.
    reposition_nudge_mps: float = 1.0
    # Default OFF (2026-09-03): the shield held a bicycle 2.4-3.9 m off the
    # ego's nose for 25 s at 2026-04-28's bus stop (in the recording it had
    # passed and was 100 m on) and forced 358 NPC stops in one run. A frozen
    # phantom is a model input the vehicle never sees; a verbatim replay's
    # drive-through contacts are at least reported honestly.
    # Replayed NPCs brake along their own recorded path instead of driving
    # through the ego, when bounded deceleration (npc_reaction_decel) would let a
    # real driver stop; they continue, delayed, once the ego is clear. Off
    # restores verbatim open-loop replay.
    npc_reaction: bool = False
    # Default 0 (off, 2026-09-03): the escape is a speed nudge, i.e. the tool
    # moving the ego; with the fixer off the only stubs are obstacle_stop's,
    # and nudging past a real obstacle guard is not a measurement.
    # Live detection of the StopPointFixer hold: near-zero speed plus nothing but
    # 3-point stop stubs on /planning/trajectory. Measured holds ran 40-92 s and
    # each cost ~60 s (30 s stall detection + a 5 m teleport that re-enters the
    # trap); the signature is unambiguous after a few seconds, and the escape is
    # an in-place speed nudge, not a teleport. 0 disables detection.
    stack_hold_detect_sec: float = 0.0
    # Re-sync (2026-09-10). The recording is the timeline: when the scene has
    # run this many seconds past the ego's own position (the ego stood where
    # the recorded vehicle drove, or drove slower), the segment ends and the
    # ego is set back onto the recording — placed at the recorded pose and
    # driven along the recorded track for the model's history window (3.5 s),
    # then handed back at the recorded speed. The NPCs are never touched. Each
    # re-sync is a divergence event (resyncs.json, the resync metric). 0 = off.
    resync_lag_sec: float = 0.0
    # Leg-window evaluation (2026-09-10, window_eval.py), the default since
    # the operator adopted it the same day. The recording is cut at the
    # recorded departures into the route's legs; each leg starts from the
    # recorded state without objects, the objects are revealed at the recorded
    # departure as the ego sets off, and from then on the scene follows the
    # ego's position (the recorded clock's rules 1-3). False restores the
    # continuous stall / nudge / re-sync drive loop.
    window_eval: bool = True
    # Planned cuts (2026-11-10, window_eval.find_cuts): "auto" also cuts each
    # leg where the recorded vehicle stood 3 s or more, at its departure, so a
    # model late at one junction still drives the next one from the recorded
    # state; "off" cuts at the leg goals only (before 2026-11-10).
    leg_cuts: str = "auto"
    # Diagnosis only: False replays no objects at all (lights and indicators
    # stay), to see how the model drives an empty road.
    scene_objects: bool = True
    # How long engage may wait for the planner's first published trajectory. A
    # fresh model builds its TensorRT engine here (minutes); zero disables the
    # gate.
    planner_warmup_timeout_sec: float = 900.0
    # Overlay-only switch for the StopPointFixer stub rules. Default ON
    # (rules off) since 2026-09-10, operator's decision: the fixer's job was
    # the pull-away, and in leg-window evaluation the ego is handed over on
    # the recorded motion, so the gate only masked the model. Measured on
    # 2026-08-24 at the y=9864 junction: the model's raw plan at green is the
    # same as the recorded shadow model's (first second 0 m/s, ~1 m/s at 4 s);
    # with the rules on that plan became a stop stub and the ego never left
    # (18 s), with them off it left with the recording. What remains with
    # them off — a brake/gas hiccup at some departures — is the model's own
    # and shows in stop_go. False restores the fixer as the vehicle has it.
    disable_long_stop_fixer: bool = True
    # Overlay-only: make the vehicle description's own acc/steer delays
    # effective (DELAY_STEER_ACC_GEARED with the description's numbers). The
    # shipped IDEAL model ignores them; with it the optimizer's 0.5 m/s engage
    # floor turned every wait into 1.0-1.6 m/s creep bursts (2026-04-28: 84 s
    # of stop-go approaching a bus stop). No invented values are written.
    match_vehicle_dynamics: bool = True
    # Overlay-only: the model reads its turn-indicator input from ONE merged
    # stream published by the reproducer (recorded signal when the recording
    # has one, psim's echo of the model's own command otherwise) instead of
    # /vehicle/status/turn_indicators_status, where psim and the replay were
    # two writers interleaving LEFT/DISABLE at ~40 Hz on the model's input.
    indicator_merge: bool = True
    # The final goal of a bag replay is where the recording ends; reaching its
    # area (20 m) ends the run without requiring the terminal park. False
    # restores strict stop-at-goal semantics for the last leg.
    final_drive_through_arrival: bool = True
    npc_reaction_decel: float = 3.0
    # Multi-goal bag replay (bus stop → arrive → next goal). Off = single start→end route.
    # "auto" (default) decides per bag: a recording that stops at intermediate points
    # needs each of them as a route goal, or the ego stops there on the planner's own
    # decision, gets no ARRIVED, and is recorded as stuck. A single stop-to-stop cut
    # has no intermediate stop and must stay single-goal, or leg extraction finds
    # nothing and the job fails outright. Which one a bag needs is readable from the
    # bag, so it should not have to be configured. True/False force the choice.
    multi_goal: bool | str = "auto"
    multi_goal_source: str = "auto"  # auto | goal_topic | stop_segments | stop_points
    multi_goal_stop_speed_mps: float = 0.2
    multi_goal_stop_min_sec: float = 8.0
    multi_goal_min_spacing_m: float = 15.0
    multi_goal_stop_order: list[str] | None = None
    stop_points_csv: str = "stop_points.csv"
    # How far a map stop point may sit from the bag's own start/goal and still be
    # substituted for it when routing fails. Negative = no limit, 0 = never substitute.
    # Unlimited is safe now that metrics follow the accepted goal (route_goals.json),
    # so a relocated run is scored where the ego actually drove; the distance moved is
    # recorded as goal_substituted_m.
    stop_fallback_max_dist_m: float = -1.0
    # Wall-clock cap on one route-setting call, recovery ladder included.
    # route_setup_timeout_sec bounds a single attempt; the ladder multiplies it
    # (attempts x backends x backoff steps x stop fallbacks), and one unroutable leg
    # goal was measured burning 581 s inside a single advance_route — ten minutes of
    # an evaluation run spent re-asking a question the mission planner had already
    # answered. 0 or negative disables the cap.
    route_total_timeout_sec: float = 180.0
    # Metres to pull the goal back along its lane when the mission planner rejects
    # it. A goal sampled where the vehicle actually parked can still fail Autoware's
    # goal-footprint-inside-lane check by centimetres. None = route_setup's default,
    # [] = disabled.
    goal_backoff_steps_m: list[float] | None = None
    # When stuck (near-zero speed), publish left/right blinker to DP input to try
    # triggering a new trajectory (experimental recovery for lane-change stalls).
    # Off by default: /vehicle/status/turn_indicators_status is produced by the
    # vehicle interface on a real car, so writing it feeds the planner a signal that
    # cannot occur on the road. Useful as a diagnostic probe, not in a scoring run.
    # Any run that uses it is counted and reported as assisted.
    stuck_blinker_nudge: bool = False
    # Speed at or below which the ego counts as stalled, for the give-up budget and
    # the near-goal arrival test. Matches Thresholds.stuck_speed_mps on purpose: the
    # run must not abandon a job for stalling at a speed the metric then scores as
    # moving. stuck_blinker_speed_mps stays separate — the nudge may fire earlier,
    # since triggering it costs nothing but a blinker.
    stall_speed_mps: float = 0.05
    stuck_blinker_speed_mps: float = 0.15
    stuck_blinker_trigger_sec: float = 12.0
    stuck_blinker_hold_sec: float = 3.0
    stuck_blinker_cooldown_sec: float = 25.0
    # Give up on a leg once the ego has held near-zero speed this long in total.
    # Counted across repositions, so it applies whether or not forward-shift is on.
    stuck_abort_sec: float = 90.0
    # While stopped (red light / lead vehicle), shift ego forward and continue.
    # trigger_sec: how long to wait stopped before each shift. forward_m: 0 disables.
    # Default OFF since the fixer's stub rules went sim-only-disabled: a
    # standing ego mid-route is now the model's own hesitation, which is the
    # thing being measured — teleporting 5 m every trigger both hid its
    # duration and put a visible lurch in the video (3x at one junction on
    # 2026-04-28). Budgets, the yield freeze and the near-goal completion
    # already keep a run from hanging forever.
    stuck_reposition_trigger_sec: float = 30.0
    stuck_reposition_forward_m: float = 0.0
    # 0 = never reposition, N>0 = at most N shifts, negative = unlimited.
    stuck_reposition_max_count: int = 3
    # Mission planner can remain SET when the bus stops just short of ARRIVED.
    multi_goal_arrival_tolerance_m: float = 5.0
    # Cap each multi-goal ARRIVED wait (full bag_duration is only the overall budget).
    multi_goal_leg_timeout_sec: float = 300.0
    # Low-CPU web live view: sample ego pose → live_drive.json for dashboard.
    live_web_monitor: bool = True
    live_web_sample_sec: float = 2.0
    live_drive_root: Path | None = None  # usually results_root (next to dashboard.html)
    # Scenario Simulator v2
    architecture_type: str = "awf/universe/20250130"
    scenario_timeout_sec: float = 300.0
    scenario_record_warmup_sec: float = 15.0
    dry_run: bool = False
    skip_route_setup: bool = False
    rviz: bool = False
    planning_setting: str = "diffusion_planner"


# Every pipeline-YAML key that reaches a job or its post-processing. run_all.py and
# run_orchestrator.py both read their config through this one list: a key missing
# from it is silently ignored, which is how four knobs (stack_hold_detect_sec,
# reposition_nudge_mps, npc_reaction, npc_reaction_decel) once spent several
# passes pinned to their defaults while the yaml said otherwise, and how
# indicator_merge sat in run_all's copy of the list without ever being forwarded.
PIPELINE_KNOBS: tuple[str, ...] = (
    # JobConfig fields
    "mode",
    "vehicle_model",
    "sensor_model",
    "vehicle_id",
    "end_condition",
    "bag_duration_margin_sec",
    "route_timeout_sec",
    "psim_startup_sec",
    "route_setup_timeout_sec",
    "auto_engage_timeout_sec",
    "perception_warmup_sec",
    "perception_ready_timeout_sec",
    "perception_ready_min_objects",
    "perception_ready_stable_sec",
    "require_perception_ready",
    "bag_load_base_sec",
    "bag_load_duration_factor",
    "bag_load_timeout_cap_sec",
    "gpu_min_free_mib",
    "gpu_wait_timeout_sec",
    "multi_goal",
    "multi_goal_source",
    "multi_goal_stop_speed_mps",
    "multi_goal_stop_min_sec",
    "multi_goal_min_spacing_m",
    "multi_goal_stop_order",
    "stop_points_csv",
    "goal_backoff_steps_m",
    "stop_fallback_max_dist_m",
    "route_total_timeout_sec",
    "stuck_blinker_nudge",
    "stall_speed_mps",
    "stuck_blinker_speed_mps",
    "stuck_blinker_trigger_sec",
    "stuck_blinker_hold_sec",
    "stuck_blinker_cooldown_sec",
    "stack_hold_detect_sec",
    "resync_lag_sec",
    "window_eval",
    "leg_cuts",
    "scene_objects",
    "planner_warmup_timeout_sec",
    "disable_long_stop_fixer",
    "match_vehicle_dynamics",
    "indicator_merge",
    "final_drive_through_arrival",
    "reposition_nudge_mps",
    "npc_reaction",
    "npc_reaction_decel",
    "stuck_abort_sec",
    "stuck_reposition_trigger_sec",
    "stuck_reposition_forward_m",
    "stuck_reposition_max_count",
    "multi_goal_arrival_tolerance_m",
    "multi_goal_leg_timeout_sec",
    "live_web_monitor",
    "live_web_sample_sec",
    "architecture_type",
    "scenario_timeout_sec",
    "scenario_record_warmup_sec",
    "planning_setting",
    "rviz",
    # post-processing (run_orchestrator)
    "render_video_on_failure",
    "render_workers",
    "video_fps",
    "video_sample_dt",
    "video_view_frame",
    "video_view_range_m",
    "show_planning_factors",
    "thresholds",
    "topics_yaml",
    "vehicle_info_yaml",
)
