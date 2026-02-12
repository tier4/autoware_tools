# Override Event Extractor

A tool for extracting override events from Autoware rosbag recordings. This tool processes rosbags in parallel, detects when the vehicle transitions to manual control mode, and extracts those segments into separate MCAP files.

## Features

- Parallel processing of multiple rosbags
- Automatic detection of override events via ControlModeReport messages
- Configurable time margins around override events
- Topic filtering to reduce output file size
- Per-rosbag and batch processing summaries in JSON format
- MCAP output format

## Building

```bash
cd /path/to/autoware/workspace
colcon build --packages-select autoware_override_extractor --symlink-install
source install/setup.bash
```

## Usage

### Using Launch File (Recommended)

```bash
ros2 launch autoware_override_extractor override_extractor.launch.py \
  input_dir:=/data/rosbags \
  output_dir:=/data/override_segments
```

### Using ros2 run

```bash
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/override_segments
```

### Advanced Options

**With launch file:**
```bash
ros2 launch autoware_override_extractor override_extractor.launch.py \
  input_dir:=/data/rosbags \
  output_dir:=/data/override_segments \
  config_file:=/path/to/custom_config.yaml
```

**With command-line arguments:**
```bash
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/override_segments \
  --threads 8 \
  --recursive \
  --min-duration 1.0 \
  --pre-margin 2.0 \
  --post-margin 15.0
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input-dir` | Input directory containing rosbags | **Required** |
| `--output-dir` | Output directory for extracted segments | **Required** |
| `--threads` | Number of parallel threads | Auto-detect |
| `--recursive` | Scan subdirectories recursively | false |
| `--min-duration` | Minimum override duration in seconds | 0.5 |
| `--no-filter-brief` | Disable filtering of brief overrides | false |
| `--topics` | Comma-separated list of topics to preserve | See config |
| `--pre-margin` | Time before override start (seconds) | 1.0 |
| `--post-margin` | Time after override end (seconds) | 10.0 |

## Output Structure

```
output_directory/
├── rosbag1_name/
│   ├── rosbag1_name_or_0.mcap
│   ├── rosbag1_name_or_1.mcap
│   └── summary.json
├── rosbag2_name/
│   ├── rosbag2_name_or_0.mcap
│   └── summary.json
└── batch_summary.json
```

## Override Detection

The tool detects override events by monitoring `/vehicle/status/control_mode` messages:
- **Override Start**: Transition from AUTONOMOUS (1) → MANUAL (4)
- **Override End**: Transition from MANUAL (4) → AUTONOMOUS (1)

Time margins are applied to capture context:
- **Pre-margin**: 1 second before override start
- **Post-margin**: 10 seconds after override end

## Configuration

Edit `config/override_extractor.param.yaml` to customize:
- Control mode topic and values
- Time margins
- Minimum override duration
- List of topics to preserve
- Thread count

## Examples

### Process all rosbags in a directory

```bash
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/2024_recordings \
  --output-dir /data/overrides \
  --recursive
```

### Custom topic list

```bash
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/output \
  --topics "/tf,/localization/kinematic_state,/planning/trajectory"
```

### Filter out very brief overrides

```bash
ros2 run autoware_override_extractor autoware_override_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/output \
  --min-duration 2.0
```

## Summary Files

### Per-Bag Summary (`summary.json`)

```json
{
  "input_bag": "/data/rosbags/recording_2024_01_15.mcap",
  "success": true,
  "num_overrides_found": 2,
  "override_events": [
    {
      "index": 0,
      "raw_start_time_ns": 1705315200000000000,
      "raw_end_time_ns": 1705315205000000000,
      "raw_duration_sec": 5.0,
      "extended_start_time_ns": 1705315199000000000,
      "extended_end_time_ns": 1705315215000000000,
      "extended_duration_sec": 16.0,
      "output_file": "recording_2024_01_15_or_0.mcap"
    }
  ]
}
```

### Batch Summary (`batch_summary.json`)

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

## Performance

- **Parallelization**: Processes multiple rosbags simultaneously
- **Thread Count**: Automatically detected based on CPU cores
- **I/O Efficiency**: MCAP format optimized for sequential writes
- **Memory**: Processes one rosbag at a time per thread

## Troubleshooting

### No overrides found

- Verify the control mode topic exists: `/vehicle/status/control_mode`
- Check that ControlModeReport messages are present in the rosbag
- Confirm mode values: MANUAL=4, AUTONOMOUS=1

### Missing topics in output

- Verify topic names match exactly (including leading `/`)
- Check the preserved topics list in configuration
- Ensure topics exist in the input rosbag

### Performance issues

- Reduce thread count if I/O is bottlenecked
- Use SSD storage for both input and output
- Process rosbags in smaller batches

## See Also

- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Detailed implementation guide
- Autoware vehicle messages: `autoware_vehicle_msgs/msg/ControlModeReport.msg`
