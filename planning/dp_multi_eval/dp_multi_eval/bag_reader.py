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
class BagSeries:
    ego: list[EgoSample] = field(default_factory=list)
    objects: list[ObjectSample] = field(default_factory=list)
    trajectories: list[TrajectorySample] = field(default_factory=list)
    velocity_mps: list[tuple[float, float]] = field(default_factory=list)  # (t, speed)

    @property
    def duration_sec(self) -> float:
        if not self.ego:
            return 0.0
        return self.ego[-1].stamp_sec - self.ego[0].stamp_sec


def open_bag_reader(bag_path: Path):
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

    bag_path = bag_path.expanduser().resolve()
    uri = str(bag_path)
    if bag_path.is_file():
        uri = str(bag_path.parent)

    storage_id = "sqlite3"
    meta = Path(uri) / "metadata.yaml"
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


def iter_topic_messages(bag_path: Path, topics: list[str]) -> Iterator[tuple[str, float, Any]]:
    """Yield (topic, stamp_sec, deserialized_msg) for selected topics."""
    from rosbag2_py import StorageFilter

    reader = open_bag_reader(bag_path)
    type_map = _type_map(reader)
    wanted = [t for t in topics if t in type_map]
    if not wanted:
        return
    reader.set_filter(StorageFilter(topics=wanted))

    while reader.has_next():
        topic, data, _t = reader.read_next()
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
    object_sample_dt: float | None = None,
    trajectory_sample_dt: float | None = None,
    skip_zero_size_objects: bool = True,
) -> BagSeries:
    series = BagSeries()
    topics = [ego_topic, objects_topic, velocity_topic]
    if trajectory_topic:
        topics.append(trajectory_topic)

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

    for topic, stamp, msg in iter_topic_messages(bag_path, topics):
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
        elif topic == objects_topic:
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

    series.ego.sort(key=lambda s: s.stamp_sec)
    series.objects.sort(key=lambda s: s.stamp_sec)
    series.trajectories.sort(key=lambda s: s.stamp_sec)
    series.velocity_mps.sort(key=lambda s: s[0])
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
