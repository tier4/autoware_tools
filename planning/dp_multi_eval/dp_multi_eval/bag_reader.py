"""Shared rosbag2 reading for metrics and offline rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quat(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def uuid_to_str(uuid_msg: Any) -> str:
    return "".join(f"{byte:02x}" for byte in uuid_msg.uuid)


@dataclass
class EgoSample:
    stamp_sec: float
    x: float
    y: float
    z: float
    yaw_rad: float
    vx: float
    vy: float
    speed_mps: float


@dataclass
class ObjectSample:
    stamp_sec: float
    object_id: str
    label: str
    x: float
    y: float
    yaw_rad: float
    length_m: float
    width_m: float


@dataclass
class TrajectorySample:
    stamp_sec: float
    points: list[tuple[float, float]]  # (x, y) in map frame


@dataclass
class VirtualWallSample:
    """One control-point pose from a PlanningFactorArray (map frame)."""

    stamp_sec: float
    source: str  # short module / topic leaf name
    x: float
    y: float
    yaw_rad: float
    behavior: int
    detail: str = ""


# TurnIndicatorsReport / Command: DISABLE=1, ENABLE_LEFT=2, ENABLE_RIGHT=3
TURN_DISABLE = 1
TURN_LEFT = 2
TURN_RIGHT = 3


@dataclass
class TurnIndicatorSample:
    stamp_sec: float
    state: int  # TURN_* 
    source: str  # "status" | "cmd"


@dataclass
class BagSeries:
    ego: list[EgoSample] = field(default_factory=list)
    objects: list[ObjectSample] = field(default_factory=list)
    trajectories: list[TrajectorySample] = field(default_factory=list)
    velocity_mps: list[tuple[float, float]] = field(default_factory=list)  # (t, speed)
    virtual_walls: list[VirtualWallSample] = field(default_factory=list)
    turn_indicators: list[TurnIndicatorSample] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        if not self.ego:
            return 0.0
        return self.ego[-1].stamp_sec - self.ego[0].stamp_sec


_BEHAVIOR_NAMES = {
    0: "UNKNOWN",
    1: "NONE",
    2: "SLOW_DOWN",
    3: "STOP",
    4: "SHIFT_LEFT",
    5: "SHIFT_RIGHT",
    6: "TURN_LEFT",
    7: "TURN_RIGHT",
}


def planning_factor_source_name(topic: str) -> str:
    return topic.rstrip("/").split("/")[-1]


def rebuild_bag_metadata(bag_dir: Path) -> bool:
    """Recreate a missing metadata.yaml from the .db3 files themselves.

    rosbag2 writes metadata.yaml only when the recorder shuts down cleanly, so a
    recorder that was killed leaves gigabytes of perfectly good messages that the
    reader refuses to open at all ("No storage could be initialized from the
    inputs"). Everything the reader needs is already in the database: the topics
    table carries names, types, serialization format and QoS, and the messages
    table carries the timestamps and counts. Losing a whole run because one small
    text file is absent is not an acceptable failure mode for a harness whose only
    product is recordings.
    """
    import sqlite3

    import yaml

    db_files = sorted(bag_dir.glob("*.db3"))
    if not db_files:
        return False

    topics: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    total = 0
    t_min: int | None = None
    t_max = 0
    for db in db_files:
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            rows = con.execute(
                "SELECT id,name,type,serialization_format,offered_qos_profiles FROM topics"
            ).fetchall()
        except sqlite3.Error:
            # Older/newer schemas may lack the QoS column.
            try:
                rows = [
                    (*r, "")
                    for r in con.execute(
                        "SELECT id,name,type,serialization_format FROM topics"
                    ).fetchall()
                ]
            except sqlite3.Error:
                con.close()
                continue
        for tid, name, ttype, fmt, qos in rows:
            count, lo, hi = con.execute(
                "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages WHERE topic_id=?",
                (tid,),
            ).fetchone()
            count = int(count or 0)
            entry = topics.setdefault(
                name,
                {
                    "topic_metadata": {
                        "name": name,
                        "type": ttype,
                        "serialization_format": fmt or "cdr",
                        "offered_qos_profiles": qos or "",
                    },
                    "message_count": 0,
                },
            )
            entry["message_count"] += count
            total += count
            if lo is not None:
                t_min = lo if t_min is None else min(t_min, lo)
                t_max = max(t_max, hi or 0)
        f_count, f_lo, f_hi = con.execute(
            "SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages"
        ).fetchone()
        files.append(
            {
                "path": db.name,
                "starting_time": {"nanoseconds_since_epoch": int(f_lo or 0)},
                "duration": {"nanoseconds": int((f_hi or 0) - (f_lo or 0))},
                "message_count": int(f_count or 0),
            }
        )
        con.close()

    if not topics:
        return False

    t_min = t_min or 0
    payload = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "relative_file_paths": [f.name for f in db_files],
            "duration": {"nanoseconds": max(0, t_max - t_min)},
            "starting_time": {"nanoseconds_since_epoch": t_min},
            "message_count": total,
            "topics_with_message_count": list(topics.values()),
            "compression_format": "",
            "compression_mode": "",
            # rosbag2 rejects the file outright without this block.
            "files": files,
        }
    }
    (bag_dir / "metadata.yaml").write_text(
        yaml.safe_dump(payload, default_flow_style=False), encoding="utf-8"
    )
    return True


def open_bag_reader(bag_path: Path):
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

    bag_path = bag_path.expanduser().resolve()
    uri = str(bag_path)
    if bag_path.is_file():
        uri = str(bag_path.parent)

    storage_id = "sqlite3"
    meta = Path(uri) / "metadata.yaml"
    if not meta.is_file() and rebuild_bag_metadata(Path(uri)):
        print(f"[warn] {uri}: metadata.yaml was missing (recorder did not exit "
              "cleanly) — rebuilt it from the .db3 files")
    if meta.is_file():
        try:
            import yaml

            raw = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            info = raw.get("rosbag2_bagfile_information") or raw
            storage_id = str(info.get("storage_identifier", storage_id))
        except Exception:  # noqa: BLE001
            pass

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=uri, storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def _type_map(reader) -> dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def iter_topic_messages(
    bag_path: Path, topics: list[str], sample_dt: dict[str, float] | None = None
) -> Iterator[tuple[str, float, Any]]:
    """Yield (topic, stamp_sec, deserialized_msg) for selected topics.

    ``sample_dt`` thins a topic to one message per interval using the storage
    timestamp, so skipped messages are never deserialized. Deserializing was
    the whole cost of a load: 48k messages, 50 s of 54 s on 2026-04-28's
    output bag, most of them trajectories and objects the caller then dropped.
    """
    from rosbag2_py import StorageFilter

    reader = open_bag_reader(bag_path)
    type_map = _type_map(reader)
    wanted = [t for t in topics if t in type_map]
    if not wanted:
        return
    reader.set_filter(StorageFilter(topics=wanted))
    interval_ns = {t: int(dt * 1e9) for t, dt in (sample_dt or {}).items() if dt}
    next_ns: dict[str, int] = {}

    while reader.has_next():
        topic, data, _t = reader.read_next()
        step = interval_ns.get(topic)
        if step:
            if _t < next_ns.get(topic, 0):
                continue
            next_ns[topic] = _t + step
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        stamp = _message_stamp_sec(msg, fallback_ns=_t)
        yield topic, stamp, msg


def _message_stamp_sec(msg: Any, fallback_ns: int) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return stamp_to_sec(msg.header.stamp)
    if hasattr(msg, "stamp"):
        return stamp_to_sec(msg.stamp)
    return float(fallback_ns) * 1e-9


def load_bag_series(
    bag_path: Path,
    *,
    ego_topic: str = "/localization/kinematic_state",
    objects_topic: str = "/perception/object_recognition/tracking/objects",
    velocity_topic: str = "/vehicle/status/velocity_status",
    trajectory_topic: str | None = "/planning/trajectory",
    planning_factor_topics: list[str] | None = None,
    turn_indicators_status_topic: str | None = "/vehicle/status/turn_indicators_status",
    turn_indicators_cmd_topic: str | None = "/planning/turn_indicators_cmd",
    object_sample_dt: float | None = None,
    trajectory_sample_dt: float | None = None,
    factor_sample_dt: float | None = None,
    skip_zero_size_objects: bool = True,
) -> BagSeries:
    series = BagSeries()
    factor_topics = list(planning_factor_topics or [])
    # Objects may sit under the throttled name (current) or the raw one (older bags).
    objects_names = [t for t in (objects_topic, "/dp_multi_eval/tracking/objects_throttled",
                                 "/perception/object_recognition/tracking/objects") if t]
    seen: set[str] = set()
    objects_names = [t for t in objects_names if not (t in seen or seen.add(t))]
    topics = [ego_topic, *objects_names, velocity_topic]
    if trajectory_topic:
        topics.append(trajectory_topic)
    topics.extend(factor_topics)
    if turn_indicators_status_topic:
        topics.append(turn_indicators_status_topic)
    if turn_indicators_cmd_topic:
        topics.append(turn_indicators_cmd_topic)

    label_names = {
        0: "UNKNOWN",
        1: "CAR",
        2: "TRUCK",
        3: "BUS",
        4: "TRAILER",
        5: "MOTORCYCLE",
        6: "BICYCLE",
        7: "PEDESTRIAN",
    }

    next_object_msg_t = -float("inf")
    next_traj_msg_t = -float("inf")
    next_factor_msg_t: dict[str, float] = {t: -float("inf") for t in factor_topics}

    sample_dt: dict[str, float] = {}
    for name in objects_names:
        if object_sample_dt:
            sample_dt[name] = float(object_sample_dt)
    if trajectory_topic and trajectory_sample_dt:
        sample_dt[trajectory_topic] = float(trajectory_sample_dt)
    for name in factor_topics:
        if factor_sample_dt:
            sample_dt[name] = float(factor_sample_dt)
    for topic, stamp, msg in iter_topic_messages(bag_path, topics, sample_dt):
        if topic == ego_topic:
            pose = msg.pose.pose
            twist = msg.twist.twist.linear
            speed = math.hypot(twist.x, twist.y)
            series.ego.append(
                EgoSample(
                    stamp_sec=stamp,
                    x=pose.position.x,
                    y=pose.position.y,
                    z=pose.position.z,
                    yaw_rad=yaw_from_quat(pose.orientation),
                    vx=twist.x,
                    vy=twist.y,
                    speed_mps=speed,
                )
            )
        elif topic == velocity_topic:
            # autoware_vehicle_msgs/VelocityReport or similar
            long_v = getattr(msg, "longitudinal_velocity", None)
            if long_v is None and hasattr(msg, "twist"):
                long_v = msg.twist.linear.x
            if long_v is None:
                continue
            series.velocity_mps.append((stamp, abs(float(long_v))))
        elif topic in objects_names:
            if object_sample_dt and stamp + 1e-9 < next_object_msg_t:
                continue
            if object_sample_dt:
                next_object_msg_t = stamp + object_sample_dt
            for obj in getattr(msg, "objects", []):
                pose = obj.kinematics.pose_with_covariance.pose
                shape = obj.shape.dimensions
                length_m = float(shape.x)
                width_m = float(shape.y)
                if skip_zero_size_objects and (length_m <= 0.0 or width_m <= 0.0):
                    continue
                classifications = getattr(obj, "classification", None) or getattr(
                    obj, "classification", []
                )
                # TrackedObjects uses classification list
                cls_list = getattr(obj, "classification", [])
                if cls_list:
                    best = max(cls_list, key=lambda c: c.probability)
                    label = label_names.get(best.label, f"LABEL_{best.label}")
                else:
                    label = "UNKNOWN"
                oid = uuid_to_str(obj.object_id) if hasattr(obj, "object_id") else "unknown"
                series.objects.append(
                    ObjectSample(
                        stamp_sec=stamp,
                        object_id=oid,
                        label=label,
                        x=pose.position.x,
                        y=pose.position.y,
                        yaw_rad=yaw_from_quat(pose.orientation),
                        length_m=length_m,
                        width_m=width_m,
                    )
                )
        elif trajectory_topic and topic == trajectory_topic:
            if trajectory_sample_dt and stamp + 1e-9 < next_traj_msg_t:
                continue
            if trajectory_sample_dt:
                next_traj_msg_t = stamp + trajectory_sample_dt
            pts = [
                (float(p.pose.position.x), float(p.pose.position.y))
                for p in getattr(msg, "points", [])
            ]
            if len(pts) >= 2:
                series.trajectories.append(TrajectorySample(stamp_sec=stamp, points=pts))
        elif topic in next_factor_msg_t:
            if factor_sample_dt and stamp + 1e-9 < next_factor_msg_t[topic]:
                continue
            if factor_sample_dt:
                next_factor_msg_t[topic] = stamp + factor_sample_dt
            source = planning_factor_source_name(topic)
            for factor in getattr(msg, "factors", []) or []:
                behavior = int(getattr(factor, "behavior", 0) or 0)
                detail = str(getattr(factor, "detail", "") or "")
                module = str(getattr(factor, "module", "") or "") or source
                for cp in getattr(factor, "control_points", []) or []:
                    pose = getattr(cp, "pose", None)
                    if pose is None:
                        continue
                    series.virtual_walls.append(
                        VirtualWallSample(
                            stamp_sec=stamp,
                            source=module,
                            x=float(pose.position.x),
                            y=float(pose.position.y),
                            yaw_rad=yaw_from_quat(pose.orientation),
                            behavior=behavior,
                            detail=detail or _BEHAVIOR_NAMES.get(behavior, str(behavior)),
                        )
                    )
        elif turn_indicators_status_topic and topic == turn_indicators_status_topic:
            state = int(getattr(msg, "report", TURN_DISABLE) or TURN_DISABLE)
            series.turn_indicators.append(
                TurnIndicatorSample(stamp_sec=stamp, state=state, source="status")
            )
        elif turn_indicators_cmd_topic and topic == turn_indicators_cmd_topic:
            state = int(getattr(msg, "command", TURN_DISABLE) or TURN_DISABLE)
            series.turn_indicators.append(
                TurnIndicatorSample(stamp_sec=stamp, state=state, source="cmd")
            )

    series.ego.sort(key=lambda s: s.stamp_sec)
    series.objects.sort(key=lambda s: s.stamp_sec)
    series.trajectories.sort(key=lambda s: s.stamp_sec)
    series.velocity_mps.sort(key=lambda s: s[0])
    series.virtual_walls.sort(key=lambda s: s.stamp_sec)
    series.turn_indicators.sort(key=lambda s: s.stamp_sec)
    return series


def downsample_ego(ego: list[EgoSample], dt: float) -> list[EgoSample]:
    if not ego or dt <= 0:
        return list(ego)
    out: list[EgoSample] = []
    next_t = ego[0].stamp_sec
    for sample in ego:
        if sample.stamp_sec + 1e-9 >= next_t:
            out.append(sample)
            next_t = sample.stamp_sec + dt
    if out and out[-1] is not ego[-1]:
        out.append(ego[-1])
    return out


REFERENCE_EGO_VERSION = "v1"


def load_reference_ego(
    bag_path: Path,
    *,
    ego_topic: str = "/localization/kinematic_state",
    velocity_topic: str = "/vehicle/status/velocity_status",
) -> list[EgoSample]:
    """The recording's ego track, read once per bag and cached.

    Metrics and the video each read it, and the recordings are 37-79 GB: the
    two reads were 90 s each of a 6.8 min post-processing step (2026-04-28).
    The track is a pure function of the recording, so it lives in the same
    cache as the route goals.
    """
    from dp_multi_eval.bag_cache import cached_json

    bag_path = Path(bag_path).expanduser().resolve()

    def compute() -> list[list[float]]:
        series = load_bag_series(
            bag_path,
            ego_topic=ego_topic,
            objects_topic=None,
            velocity_topic=velocity_topic,
            trajectory_topic=None,
            turn_indicators_status_topic=None,
            turn_indicators_cmd_topic=None,
        )
        return [[e.stamp_sec, e.x, e.y, e.z, e.yaw_rad, e.vx, e.vy, e.speed_mps] for e in series.ego]

    rows = cached_json(bag_path, "reference_ego", f"{REFERENCE_EGO_VERSION}|{ego_topic}|{velocity_topic}", compute)
    return [EgoSample(*row) for row in rows]
