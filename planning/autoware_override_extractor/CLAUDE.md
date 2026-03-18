# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

**autoware_override_extractor** extracts manual override segments from Autoware rosbags. It detects transitions between AUTONOMOUS (mode=1) and MANUAL (mode=4) control via `/vehicle/status/control_mode` messages, then splits rosbags into separate MCAP files with configurable time margins.

**Key Capabilities:**
- Parallel processing of multiple rosbags using thread pools
- Route message caching and replication across override segments
- Topic filtering to reduce output file size
- Automatic merging of overlapping override events
- JSON summaries per-bag and batch-wide

## Building and Running

### Build Commands

```bash
# From workspace root (e.g., /path/to/pilot-auto.x2)
colcon build --packages-select autoware_override_extractor --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Running the Tool

**Using launch file (recommended):**
```bash
# With launch file (loads config/override_extractor.param.yaml by default)
ros2 launch autoware_override_extractor override_extractor.launch.py \
  input_dir:=/data/rosbags \
  output_dir:=/data/override_segments

# With custom config file
ros2 launch autoware_override_extractor override_extractor.launch.py \
  input_dir:=/data/rosbags \
  output_dir:=/data/override_segments \
  config_file:=/path/to/custom_config.yaml
```

**Using ros2 run with command-line args:**
```bash
# Basic usage
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/override_segments

# Override specific parameters
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/override_segments \
  --threads 8 \
  --recursive \
  --min-duration 1.0 \
  --pre-margin 2.0 \
  --post-margin 15.0

# Or use ROS 2 parameter syntax
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --ros-args \
  --params-file config/override_extractor.param.yaml \
  -p input_dir:=/data/rosbags \
  -p output_dir:=/data/override_segments
```

### Linting

```bash
# From workspace root
pre-commit run --files src/tools/planning/autoware_override_extractor/**/*
```

## Architecture

The codebase follows a **4-layer pipeline architecture**:

```
ParallelExecutor (parallel_executor.hpp)
  └─> BagProcessor (bag_processor.hpp)
       ├─> OverrideDetector (override_detector.hpp)  [detects override events]
       └─> BagSplitter (bag_splitter.hpp)            [extracts segments]
```

### Layer Responsibilities

1. **ParallelExecutor**: Entry point for batch processing
   - Discovers rosbags recursively in input directory
   - Extracts route messages from all bags ONCE before processing (stored in `route_cache_`)
   - Manages thread pool for parallel bag processing
   - Generates `batch_summary.json`

2. **BagProcessor**: Single-bag orchestration
   - Coordinates detector and splitter for one rosbag
   - Generates per-bag `summary.json` with override event metadata
   - Returns `ProcessResult` with success/failure status

3. **OverrideDetector**: Override event detection logic
   - Scans `/vehicle/status/control_mode` for transitions
   - Applies pre/post margins to create `extended_range`
   - Merges overlapping events after margin expansion
   - Filters events below `min_override_duration_sec`

4. **BagSplitter**: Segment extraction and MCAP writing
   - Reads input bag, writes messages in `extended_range` to output MCAP
   - Filters topics using `preserved_topics_set_` (unordered_set for O(1) lookup)
   - **ALWAYS writes route message first** if available, regardless of timestamp

### Key Data Structures (type_alias.hpp)

- **TimeRange**: Nanosecond-precision time span with overlap/merge utilities
- **OverrideEvent**: Pairs `raw_range` (actual override) with `extended_range` (with margins)
- **DetectorConfig**: Override detection parameters (margins, thresholds)
- **SplitterConfig**: Output format and topic filtering settings
- **RouteMessage**: Cached route from `/planning/mission_planning/route`

## Critical Implementation Details

### Route Message Handling

Route messages are extracted **once per rosbag series** in `ParallelExecutor::extract_route_messages()`:
- Groups rosbags by series name (e.g., `foo_0.db3`, `foo_1.db3` → series `foo`)
- Reads route from first bag in series, caches in `route_cache_`
- **Every override segment gets the route written first**, even if timestamp is out of range

**Why:** Cropped override bags often don't contain route messages (route was published before the override happened). Injecting the cached route makes these segments usable for downstream analysis tools.

### Override Detection State Machine

OverrideDetector tracks transitions between control modes:
```
AUTONOMOUS (1) → MANUAL (4) = Override START
MANUAL (4) → AUTONOMOUS (1) = Override END
```

**Edge cases handled:**
- Bag starts/ends mid-override → event clipped to bag bounds
- Multiple transitions within margin period → merged into single event
- Very brief overrides (< `min_override_duration_sec`) → filtered out

### Override Processing Pipeline

`OverrideDetector::detect()` processes events in this order:

1. **Detect transitions**: Scan ControlModeReport messages for AUTONOMOUS↔MANUAL changes
2. **Filter by raw duration**: Reject overrides shorter than `min_override_duration_sec` (line 63)
3. **Apply time margins**: Extend each event with pre/post margins to create `extended_range` (line 65)
4. **Merge overlapping**: Combine events whose extended ranges overlap (line 89)

**Key insight:** Filtering happens on the **raw** override duration, not the extended duration. This ensures brief overrides are rejected even if margins would extend them.

**Merge example:**
```
Raw events:    [====]    [====]     (2 separate overrides, both > min_duration)
With margins:  [========][========]  (extended ranges overlap)
After merge:   [================]    (single output segment)
```

### Parallel Processing

`ParallelExecutor::process_with_thread_pool()` uses `std::thread` directly:
- Thread count: auto-detected via `std::thread::hardware_concurrency()` or user-specified
- Each thread processes one rosbag at a time (no shared state except `route_cache_`)
- Results collected in pre-allocated vector with mutex protection

## Configuration

Configuration is loaded from `config/override_extractor.param.yaml` by default when using the launch file. Parameters can be overridden via:
1. Custom config file (`config_file` launch argument)
2. Launch arguments (`input_dir:=...`, `output_dir:=...`)
3. Command-line arguments (`--input-dir`, `--pre-margin`, etc.)
4. ROS 2 parameter CLI (`-p param_name:=value`)

### Key Parameters

**Detection settings:**
- `control_mode_topic`: Topic for ControlModeReport messages (default: `/vehicle/status/control_mode`)
- `override_mode_value`: Control mode value for MANUAL (default: 4)
- `autonomous_mode_value`: Control mode value for AUTONOMOUS (default: 1)

**Time margins:**
- `pre_margin`: Seconds before override start (default: 1.0)
- `post_margin`: Seconds after override end (default: 10.0)
- Rationale: 1s pre-margin captures vehicle state before override; 10s post-margin captures recovery behavior

**Filtering:**
- `filter_brief_overrides`: Filter out short overrides (default: true)
- `min_override_duration`: Minimum raw override duration in seconds (default: 0.5)

**Topic preservation:**
- `preserved_topics`: List of topics to include in output segments (see YAML for defaults)

**Execution:**
- `num_threads`: Parallel processing threads, -1 for auto-detect (default: -1)
- `recursive_scan`: Scan subdirectories recursively (default: true)

## Common Development Patterns

### Adding New Parameters

1. Add field to appropriate config struct in `type_alias.hpp`
2. Add `declare_parameter()` in `OverrideExtractorNode` constructor
3. Add `get_parameter()` in `OverrideExtractorNode::get_config()`
4. Add parameter to `config/override_extractor.param.yaml` with default value and description
5. (Optional) Add command-line argument parsing in `main()` for override support
6. (Optional) Update `print_usage()` if adding command-line support

### Adding New Preserved Topics

Update `config/override_extractor.param.yaml`:
```yaml
preserved_topics:
  - "/your/new/topic"
  - "/another/topic"
```

Or override at runtime:
```bash
ros2 launch autoware_override_extractor override_extractor.launch.py \
  input_dir:=/data \
  output_dir:=/output \
  --ros-args -p preserved_topics:="['/topic1','/topic2']"
```

### Debugging Rosbag Issues

Check these in order:
1. Bag metadata: Does `/vehicle/status/control_mode` exist?
2. Mode values: Are transitions AUTONOMOUS(1) ↔ MANUAL(4)?
3. Timestamps: Are messages in chronological order?
4. Storage plugin: Is MCAP plugin available? (`rosbag2_storage_mcap`)

## Testing Strategy

**Current state:** No unit tests implemented.

**Recommended test structure:**
```cpp
// test/test_override_detector.cpp
TEST(OverrideDetectorTest, DetectsSingleOverride) { ... }
TEST(OverrideDetectorTest, MergesOverlappingEvents) { ... }
TEST(OverrideDetectorTest, FiltersShortOverrides) { ... }

// test/test_bag_splitter.cpp
TEST(BagSplitterTest, PreservesOnlyConfiguredTopics) { ... }
TEST(BagSplitterTest, WritesRouteMessageFirst) { ... }
```

Use `rosbag2_test_common` utilities for creating test bags in memory.

## Output Files

### Directory Structure
```
output_dir/
├── rosbag1_name/
│   ├── rosbag1_name_or_0.mcap      # First override segment
│   ├── rosbag1_name_or_1.mcap      # Second override segment
│   └── summary.json                # Per-bag metadata
└── batch_summary.json              # Batch-level statistics
```

### Summary JSON Schema

**Per-bag** (`summary.json`):
```json
{
  "input_bag": "/full/path/to/input.mcap",
  "success": true,
  "num_overrides_found": 2,
  "override_events": [
    {
      "index": 0,
      "raw_start_time_ns": <int64>,
      "raw_end_time_ns": <int64>,
      "raw_duration_sec": <double>,
      "extended_start_time_ns": <int64>,
      "extended_end_time_ns": <int64>,
      "extended_duration_sec": <double>,
      "output_file": "input_or_0.mcap"
    }
  ]
}
```

**Batch** (`batch_summary.json`):
```json
{
  "input_directory": "/data/rosbags",
  "output_directory": "/data/overrides",
  "results": {
    "total_bags_processed": 25,
    "total_overrides_found": 47,
    "failed_bags": 0
  }
}
```

## Dependencies

**Core ROS2 dependencies:**
- `rosbag2_cpp`: Bag reading/writing API
- `rosbag2_storage`: Storage backend abstraction
- `autoware_vehicle_msgs`: ControlModeReport message definition

**Runtime storage plugins:**
- `rosbag2_storage_mcap`: MCAP format support (default output)
- `rosbag2_storage_default_plugins`: DB3/SQLite3 input support

## Diffusion Planner Compatibility

The default preserved topics are configured for compatibility with **Diffusion Planner** NPZ dataset creation (`parse_rosbag_for_directory.py`).

**Required topics for Diffusion Planner:**
- `/localization/kinematic_state` - Ego odometry
- `/localization/acceleration` - Ego acceleration
- `/perception/object_recognition/tracking/objects` - Tracked objects (neighbors)
- `/perception/traffic_light_recognition/traffic_signals` - Traffic light states
- `/vehicle/status/turn_indicators_status` - Turn indicator state
- `/planning/mission_planning/route` - Route (automatically injected from first bag)
- `/tf`, `/tf_static` - Transforms

**Note:** Vector map (`/map/vector_map`) is NOT recorded in rosbags by default. The Diffusion Planner data converter expects the map to be provided separately from the map directory structure.

**Workflow:**
1. Extract override segments with this tool
2. Organize output in Diffusion Planner structure:
   ```
   driving_dataset/
   ├── bag/
   │   └── date/
   │       └── time/  # <-- Your override segments go here
   └── map/
       └── date/
           └── lanelet2_map.osm
   ```
3. Run `parse_rosbag_for_directory.py` to create NPZ files

## Related Packages

- **autoware_planning_data_analyzer**: Downstream analysis of extracted segments
- **autoware_rtc_replayer**: Replay RTC events from extracted segments
- **autoware_diffusion_planner**: ML-based trajectory planner (dataset consumer)
- **override_event_extractor**: Old package name (deprecated, replaced by this package)
