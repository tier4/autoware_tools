#!/usr/bin/env bash
# Bring up Foxglove camera overlay (trajectory / virtual walls / perception)
# without launching logging_simulator.
#
# Usage:
#   ros2 run planning_debug_tools foxglove_camera_overlay.sh --bag /path/to/bag
#   # or after install path:
#   ./foxglove_camera_overlay.sh --bag /path/to/bag
#
# Options:
#   --bag PATH              rosbag2 path (required unless --no-bag)
#   --no-bag                do not play a bag (you play it yourself)
#   --camera N              camera index (default: 8 = front)
#   --vehicle-model NAME    default: lv828l
#   --sensor-model NAME     default: aip_x2_gen2
#   --vehicle-id ID         default: 6_lv828l
#   --no-vehicle-tf         skip robot_state_publisher (bag already has camera TF)
#   --no-studio             do not open Foxglove Studio GUI
#   --rate R                bag play rate (default: 1.0)
#   --help

set -euo pipefail

CAMERA=8
BAG=""
PLAY_BAG=1
LAUNCH_VEHICLE_TF=1
OPEN_STUDIO=1
VEHICLE_MODEL=lv828l
SENSOR_MODEL=aip_x2_gen2
VEHICLE_ID=6_lv828l
RATE=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT_CANDIDATES=(
  "${SCRIPT_DIR}/../config/foxglove_camera8_overlay.json"
  "${SCRIPT_DIR}/../../share/planning_debug_tools/config/foxglove_camera8_overlay.json"
)
if command -v ros2 >/dev/null 2>&1; then
  SHARE="$(ros2 pkg prefix planning_debug_tools 2>/dev/null || true)/share/planning_debug_tools/config/foxglove_camera8_overlay.json"
  LAYOUT_CANDIDATES+=("${SHARE}")
fi

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag) BAG="$2"; shift 2 ;;
    --no-bag) PLAY_BAG=0; shift ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --vehicle-model) VEHICLE_MODEL="$2"; shift 2 ;;
    --sensor-model) SENSOR_MODEL="$2"; shift 2 ;;
    --vehicle-id) VEHICLE_ID="$2"; shift 2 ;;
    --no-vehicle-tf) LAUNCH_VEHICLE_TF=0; shift ;;
    --no-studio) OPEN_STUDIO=0; shift ;;
    --rate) RATE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ "${PLAY_BAG}" -eq 1 && -z "${BAG}" ]]; then
  echo "error: --bag PATH is required (or pass --no-bag)"
  exit 1
fi

IMAGE_COMPRESSED="/sensing/camera/camera${CAMERA}/image_raw/compressed"
CAMERA_INFO="/sensing/camera/camera${CAMERA}/camera_info"
CAMERA_INFO_RELAY="/sensing/camera/camera${CAMERA}/image_raw/camera_info"

PIDS=()
cleanup() {
  echo "[foxglove_overlay] shutting down..."
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_bg() {
  echo "[foxglove_overlay] $*"
  "$@" &
  PIDS+=("$!")
}

# --- Foxglove bridge ---
start_bg ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765

# --- Sensor kit TF (camera optical frames) ---
# Must use individual_params calibration dir (vehicle_id/sensor_model), not the
# sensor_kit description default — otherwise xacro fails and no camera TF exists.
if [[ "${LAUNCH_VEHICLE_TF}" -eq 1 ]]; then
  CONFIG_DIR="$(ros2 pkg prefix individual_params)/share/individual_params/config/${VEHICLE_ID}/${SENSOR_MODEL}"
  if [[ ! -f "${CONFIG_DIR}/sensors_calibration.yaml" ]]; then
    echo "error: missing ${CONFIG_DIR}/sensors_calibration.yaml"
    echo "  check --vehicle-id / --sensor-model"
    exit 1
  fi
  start_bg ros2 launch tier4_vehicle_launch vehicle.launch.xml \
    vehicle_model:="${VEHICLE_MODEL}" \
    sensor_model:="${SENSOR_MODEL}" \
    vehicle_id:="${VEHICLE_ID}" \
    config_dir:="${CONFIG_DIR}" \
    launch_vehicle_interface:=false \
    launch_auto_vehicle_engage:=false \
    auto_vehicle_engage_param_path:=/dev/null
fi

# --- camera_info under image_raw/ for Foxglove Calibration ---
start_bg ros2 run topic_tools relay "${CAMERA_INFO}" "${CAMERA_INFO_RELAY}"

# --- Convert trajectory + predicted objects to MarkerArray ---
start_bg ros2 run planning_debug_tools foxglove_overlay_markers.py \
  --ros-args -p use_sim_time:=true

# --- Optional bag play (filtered to topics that exist in the bag) ---
if [[ "${PLAY_BAG}" -eq 1 ]]; then
  WANTED=(
    /clock
    /tf
    /tf_static
    "${IMAGE_COMPRESSED}"
    "${CAMERA_INFO}"
    /localization/kinematic_state
    /map/vector_map
    /map/vector_map_marker
    /map/pointcloud_map
    /perception/object_recognition/objects
    /perception/object_recognition/detection/objects
    /perception/object_recognition/tracking/objects
    /planning/trajectory
    /planning/scenario_planning/lane_driving/trajectory
    /planning/planning_factors/modifier_obstacle_stop
    /planning/planning_factors/diffusion_planner
    /vehicle/status/control_mode
    /api/operation_mode/state
    /control/trajectory_follower/controller_node_exe/virtual_wall
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/blind_spot
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/crosswalk
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/detection_area
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/intersection
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/merge_from_private
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/no_stopping_area
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/occlusion_spot
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/run_out
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/stop_line
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/traffic_light
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/virtual_traffic_light
    /planning/scenario_planning/lane_driving/behavior_planning/behavior_velocity_planner/virtual_wall/walkway
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/obstacle_stop/virtual_walls
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/obstacle_cruise/virtual_walls
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/obstacle_slow_down/virtual_walls
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/suitable_stop/virtual_walls
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/out_of_lane/virtual_walls
    /planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/dynamic_obstacle_stop/virtual_walls
  )

  mapfile -t BAG_TOPICS < <(ros2 bag info "${BAG}" 2>/dev/null | sed -n 's/^[[:space:]]*Topic:[[:space:]]*//p' | sort -u)
  PLAY_TOPICS=()
  for t in "${WANTED[@]}"; do
    for b in "${BAG_TOPICS[@]:-}"; do
      if [[ "${t}" == "${b}" ]]; then
        PLAY_TOPICS+=("${t}")
        break
      fi
    done
  done

  if [[ "${#PLAY_TOPICS[@]}" -eq 0 ]]; then
    echo "[foxglove_overlay] warning: no wanted topics found in bag; playing entire bag"
    start_bg ros2 bag play "${BAG}" --clock --rate "${RATE}"
  else
    echo "[foxglove_overlay] playing ${#PLAY_TOPICS[@]} topics from bag"
    start_bg ros2 bag play "${BAG}" --clock --rate "${RATE}" --topics "${PLAY_TOPICS[@]}"
  fi
fi

sleep 2

# --- Open Foxglove Studio ---
LAYOUT=""
for cand in "${LAYOUT_CANDIDATES[@]}"; do
  if [[ -f "${cand}" ]]; then
    LAYOUT="${cand}"
    break
  fi
done

if [[ "${OPEN_STUDIO}" -eq 1 ]]; then
  if command -v foxglove-studio >/dev/null 2>&1; then
    echo "[foxglove_overlay] opening Foxglove Studio (ws://localhost:8765)"
    # Deep-link to websocket; import layout JSON via File > Open layout if needed.
    foxglove-studio "foxglove://open?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8765" >/dev/null 2>&1 &
    PIDS+=("$!")
  else
    echo "[foxglove_overlay] foxglove-studio not found; open Studio manually -> ws://localhost:8765"
  fi
fi

echo
echo "=============================================="
echo " Foxglove camera overlay ready"
echo "=============================================="
echo " Camera image : ${IMAGE_COMPRESSED}"
echo " Calibration  : ${CAMERA_INFO_RELAY}"
echo " Trajectory   : /foxglove/overlay/trajectory"
echo " Perception   : /foxglove/overlay/predicted_objects"
echo " Factors      : /foxglove/overlay/planning_factors/{modifier_obstacle_stop,diffusion_planner}"
echo " Mode HUD     : /foxglove/overlay/control_mode_2d  (fixed AUTO/MANUAL)"
echo " Bridge       : ws://localhost:8765"
if [[ -n "${LAYOUT}" ]]; then
  echo " Layout JSON  : ${LAYOUT}"
  echo "   In Foxglove: Layouts -> Import layout from file -> select the JSON above"
fi
echo
echo " Image panel Topics (enable eyes):"
echo "   /foxglove/overlay/trajectory"
echo "   /foxglove/overlay/predicted_objects"
echo "   /foxglove/overlay/planning_factors/modifier_obstacle_stop"
echo "   /foxglove/overlay/planning_factors/diffusion_planner"
echo "   /map/vector_map_marker"
echo "   virtual_wall / virtual_walls topics"
echo " Image annotations (enable):"
echo "   /foxglove/overlay/control_mode_2d"
echo
echo " Ctrl+C to stop."
echo "=============================================="

wait
