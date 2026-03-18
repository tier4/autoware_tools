# Override Event Extractor - Implementation Guide

## Overview

A parallel rosbag processing tool that detects vehicle override events and extracts them into separate MCAP files.

## Specifications

### Core Requirements
- **Input**: Directory containing rosbag files (MCAP or SQLite3 format)
- **Detection**: Monitor `ControlModeReport` messages for MANUAL mode transitions
- **Output**: MCAP files containing override event segments
- **Parallelization**: Process multiple rosbags simultaneously using thread pool

### Override Detection Rules
```
Override Event = Transition from AUTONOMOUS → MANUAL mode

ControlModeReport.mode values:
- AUTONOMOUS = 1 (normal operation)
- MANUAL = 4 (override active)
```

### Time Windows
- **Pre-margin**: -1 second before override starts
- **Post-margin**: +10 seconds after override ends
- **Minimum duration filter**: Configurable (default: 0.5 seconds)
  - Overrides shorter than this are optionally filtered out

### Topic Preservation
- User-configurable list of topics to copy
- Only specified topics are written to output MCAP files
- Reduces output file size and focuses on relevant data

### Output Structure
```
output_directory/
├── rosbag1_name/
│   ├── rosbag1_name_or_0.mcap
│   ├── rosbag1_name_or_1.mcap
│   ├── rosbag1_name_or_2.mcap
│   └── summary.json
├── rosbag2_name/
│   ├── rosbag2_name_or_0.mcap
│   └── summary.json
└── batch_summary.json
```

**Key Design**: Each rosbag gets its own subdirectory, enabling lock-free parallel processing.

### Naming Convention
```
Original: /path/to/my_recording_2024.mcap
Outputs:  output_dir/my_recording_2024/
          ├── my_recording_2024_or_0.mcap  (first override)
          ├── my_recording_2024_or_1.mcap  (second override)
          └── my_recording_2024_or_N.mcap  (Nth override)
```

---

## Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    main.cpp                             │
│  - Parse CLI arguments                                  │
│  - Load configuration                                   │
│  - Create ParallelExecutor                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              ParallelExecutor                           │
│  - Discover rosbags in input directory                  │
│  - Create thread pool                                   │
│  - Distribute rosbags to worker threads                 │
│  - Aggregate results                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (per thread)
┌─────────────────────────────────────────────────────────┐
│               BagProcessor                              │
│  - Orchestrate single rosbag processing                 │
│  - Call OverrideDetector                                │
│  - Call BagSplitter                                     │
│  - Generate per-bag summary                             │
└──────────┬─────────────────────┬────────────────────────┘
           │                     │
           ▼                     ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│  OverrideDetector    │  │     BagSplitter              │
│  - Scan for          │  │  - Read input rosbag         │
│    ControlModeReport │  │  - Filter by time ranges     │
│  - Track state       │  │  - Filter by topic list      │
│    transitions       │  │  - Write MCAP outputs        │
│  - Apply margins     │  │                              │
│  - Filter by min     │  │                              │
│    duration          │  │                              │
└──────────────────────┘  └──────────────────────────────┘
```

---

## Component Details

### 1. OverrideDetector (`override_detector.hpp/cpp`)

**Purpose**: Scan rosbag for ControlModeReport messages and identify override events.

**Key Data Structures**:
```cpp
struct TimeRange {
  rcutils_time_point_value_t start_ns;
  rcutils_time_point_value_t end_ns;

  double duration_seconds() const {
    return (end_ns - start_ns) / 1e9;
  }
};

struct OverrideEvent {
  TimeRange raw_range;        // Exact MANUAL mode period
  TimeRange extended_range;   // With pre/post margins applied
  size_t index;               // 0, 1, 2, ... for this rosbag
};
```

**Algorithm**:
```
1. Open rosbag with StorageFilter for /vehicle/status/control_mode
2. Iterate through all ControlModeReport messages
3. Track state transitions:
   - When mode changes from AUTONOMOUS(1) → MANUAL(4):
     * Record override_start_time
   - When mode changes from MANUAL(4) → AUTONOMOUS(1):
     * Record override_end_time
     * Create OverrideEvent with [start, end] range
4. Apply time margins:
   - extended_start = override_start - 1 second
   - extended_end = override_end + 10 seconds
5. Filter by minimum duration:
   - If (override_end - override_start) < min_duration:
     * Skip this event (if filter_brief_overrides = true)
6. Merge overlapping extended ranges (if any)
7. Return vector<OverrideEvent>
```

**Key Methods**:
```cpp
class OverrideDetector {
public:
  OverrideDetector(const DetectorConfig& config);

  std::vector<OverrideEvent> detect(const std::string& bag_path);

private:
  bool isOverrideActive(uint8_t mode) const;
  void applyMargins(OverrideEvent& event) const;
  std::vector<OverrideEvent> mergeOverlapping(
    const std::vector<OverrideEvent>& events) const;

  DetectorConfig config_;
};
```

---

### 2. BagSplitter (`bag_splitter.hpp/cpp`)

**Purpose**: Extract override event segments from rosbag and write to MCAP files.

**Key Operations**:
```cpp
class BagSplitter {
public:
  BagSplitter(const SplitterConfig& config);

  // Split single rosbag into multiple output files
  std::vector<std::string> split(
    const std::string& input_bag,
    const std::vector<OverrideEvent>& events,
    const std::string& output_dir
  );

private:
  // Extract one override segment
  void extractSegment(
    rosbag2_cpp::Reader& reader,
    rosbag2_cpp::Writer& writer,
    const TimeRange& range
  );

  // Check if message timestamp is within range
  bool inRange(
    rcutils_time_point_value_t timestamp,
    const TimeRange& range
  ) const;

  // Check if topic should be preserved
  bool shouldPreserveTopic(const std::string& topic_name) const;

  SplitterConfig config_;
};
```

**Algorithm**:
```
For each OverrideEvent:
  1. Create output filename: <original>_or_<index>.mcap
  2. Create rosbag2_cpp::Writer with MCAP storage
  3. Set up topic metadata (copy from input bag)
  4. Rewind input rosbag to beginning
  5. Iterate through ALL messages:
     - Check if message.timestamp in [event.start, event.end]
     - Check if message.topic in preserved_topics list
     - If both true: write message to output bag
  6. Close writer
  7. Return output filename
```

**Important Implementation Details**:
- Use `rosbag2_storage::StorageOptions` with `storage_id = "mcap"`
- Preserve topic metadata (type, serialization, QoS)
- Handle message deserialization/serialization correctly
- Efficient seeking if rosbag2 API supports it

---

### 3. BagProcessor (`bag_processor.hpp/cpp`)

**Purpose**: Orchestrate detection + splitting for a single rosbag.

**Key Data Structures**:
```cpp
struct ProcessResult {
  std::string input_bag;
  bool success;
  std::string error_message;
  size_t num_overrides_found;
  std::vector<std::string> output_files;
  std::vector<OverrideEvent> events;
};
```

**Workflow**:
```cpp
class BagProcessor {
public:
  BagProcessor(const ProcessorConfig& config);

  ProcessResult process(
    const std::string& bag_path,
    const std::string& output_dir
  );

private:
  void generateSummary(
    const ProcessResult& result,
    const std::string& output_dir
  ) const;

  ProcessorConfig config_;
  OverrideDetector detector_;
  BagSplitter splitter_;
};
```

**Algorithm**:
```
1. Validate input rosbag exists and is readable
2. Create output subdirectory: output_dir/<bag_name>/
3. Call detector.detect(bag_path) → vector<OverrideEvent>
4. If no overrides found:
   - Log info message
   - Return success with 0 overrides
5. Call splitter.split(bag_path, events, output_subdir)
6. Generate summary.json with event details
7. Return ProcessResult
```

---

### 4. ParallelExecutor (`parallel_executor.hpp/cpp`)

**Purpose**: Coordinate parallel processing of multiple rosbags.

**Key Data Structures**:
```cpp
struct ExecutorConfig {
  std::string input_dir;
  std::string output_dir;
  int num_threads;  // -1 = auto-detect
  bool recursive;   // Scan subdirectories
};

struct BatchResult {
  size_t total_bags_processed;
  size_t total_overrides_found;
  size_t failed_bags;
  std::vector<ProcessResult> results;
};
```

**Algorithm**:
```cpp
class ParallelExecutor {
public:
  ParallelExecutor(const ExecutorConfig& config);

  BatchResult execute();

private:
  std::vector<std::string> discoverRosbags() const;
  void processWithThreadPool(
    const std::vector<std::string>& bags,
    std::vector<ProcessResult>& results
  );
  void generateBatchSummary(const BatchResult& result) const;

  ExecutorConfig config_;
  BagProcessor processor_;
};
```

**Thread Pool Implementation**:
```
1. Discover all rosbag files in input_dir
   - Support both .mcap and .db3 extensions
   - Optionally recurse into subdirectories
2. Determine thread count:
   - If num_threads = -1: use std::thread::hardware_concurrency()
   - Else: use specified value
3. Create result vector (size = num_bags)
4. Use std::async or thread pool to process bags in parallel:
   - Each thread calls processor.process(bag_path, output_dir)
   - Store result in thread-safe manner (pre-allocated vector indices)
5. Wait for all threads to complete
6. Aggregate results into BatchResult
7. Generate batch_summary.json
```

**Key Point**: No mutex needed because each thread writes to its own subdirectory.

---

## Configuration

### Parameter File (`config/override_extractor.param.yaml`)

```yaml
/**:
  ros__parameters:
    # Override detection settings
    control_mode_topic: "/vehicle/status/control_mode"
    override_mode_value: 4  # MANUAL
    autonomous_mode_value: 1  # AUTONOMOUS

    # Time margins (seconds)
    pre_margin: 1.0   # Before override starts
    post_margin: 10.0 # After override ends

    # Filtering
    filter_brief_overrides: true
    min_override_duration: 0.5  # seconds

    # Topics to preserve in output MCAP files
    preserved_topics:
      - "/vehicle/status/control_mode"
      - "/localization/kinematic_state"
      - "/perception/object_recognition/objects"
      - "/planning/trajectory"
      - "/vehicle/status/steering_status"
      - "/vehicle/status/velocity_status"
      - "/tf"
      - "/tf_static"

    # Parallel processing
    num_threads: -1  # -1 = auto-detect
    recursive_scan: true

    # Output settings
    output_format: "mcap"
    storage_preset: "mcap"
```

---

## Command-Line Interface

### Arguments
```bash
ros2 run override_event_extractor override_event_extractor_node \
  --input-dir <path>          # Input directory with rosbags
  --output-dir <path>         # Output directory for extracted segments
  [--threads N]               # Number of parallel threads (default: auto)
  [--recursive]               # Scan subdirectories
  [--min-duration SECONDS]    # Minimum override duration filter
  [--no-filter-brief]         # Disable brief override filtering
  [--topics topic1,topic2]    # Override preserved topics list
```

### Example Usage
```bash
# Process all rosbags in directory with 8 threads
ros2 run override_event_extractor override_event_extractor_node \
  --input-dir /data/rosbags/2024_recordings \
  --output-dir /data/override_segments \
  --threads 8 \
  --recursive

# Process with custom topic list
ros2 run override_event_extractor override_event_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/output \
  --topics "/tf,/localization/kinematic_state,/planning/trajectory"
```

---

## Summary JSON Format

### Per-Bag Summary (`<bag_name>/summary.json`)
```json
{
  "input_bag": "/data/rosbags/recording_2024_01_15.mcap",
  "processing_timestamp": "2024-01-15T10:30:00Z",
  "success": true,
  "num_overrides_found": 3,
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
    },
    {
      "index": 1,
      "raw_start_time_ns": 1705315300000000000,
      "raw_end_time_ns": 1705315302000000000,
      "raw_duration_sec": 2.0,
      "extended_start_time_ns": 1705315299000000000,
      "extended_end_time_ns": 1705315312000000000,
      "extended_duration_sec": 13.0,
      "output_file": "recording_2024_01_15_or_1.mcap"
    }
  ]
}
```

### Batch Summary (`batch_summary.json`)
```json
{
  "processing_timestamp": "2024-01-15T10:30:00Z",
  "input_directory": "/data/rosbags/2024_recordings",
  "output_directory": "/data/override_segments",
  "configuration": {
    "num_threads": 8,
    "pre_margin_sec": 1.0,
    "post_margin_sec": 10.0,
    "min_override_duration_sec": 0.5,
    "filter_brief_overrides": true
  },
  "results": {
    "total_bags_processed": 25,
    "total_overrides_found": 47,
    "failed_bags": 2
  },
  "per_bag_summary": [
    {
      "bag_name": "recording_2024_01_15.mcap",
      "overrides_found": 3,
      "success": true
    },
    {
      "bag_name": "recording_2024_01_16.mcap",
      "overrides_found": 0,
      "success": true
    }
  ]
}
```

---

## Implementation Checklist

### Phase 1: Foundation
- [ ] Create package structure (package.xml, CMakeLists.txt)
- [ ] Create `type_alias.hpp` with ControlModeReport
- [ ] Set up configuration parameter loading
- [ ] Implement basic command-line argument parsing in `main.cpp`

### Phase 2: Override Detection
- [ ] Implement `OverrideDetector` class
- [ ] Parse ControlModeReport messages from rosbag
- [ ] State machine for AUTONOMOUS ↔ MANUAL transitions
- [ ] Apply time margins (pre: -1s, post: +10s)
- [ ] Implement minimum duration filtering
- [ ] Implement overlapping range merging
- [ ] Unit test with sample rosbag

### Phase 3: Rosbag Splitting
- [ ] Implement `BagSplitter` class
- [ ] Set up rosbag2_cpp::Writer with MCAP storage
- [ ] Implement topic filtering logic
- [ ] Implement time-range filtering logic
- [ ] Copy messages within range to output MCAP
- [ ] Preserve topic metadata correctly
- [ ] Test with real rosbag

### Phase 4: Single Bag Processing
- [ ] Implement `BagProcessor` class
- [ ] Integrate OverrideDetector + BagSplitter
- [ ] Create per-bag output subdirectory
- [ ] Generate summary.json for each rosbag
- [ ] Handle edge cases (no overrides, corrupted bags)
- [ ] Error handling and logging

### Phase 5: Parallel Execution
- [ ] Implement `ParallelExecutor` class
- [ ] Directory scanning for rosbags (with recursive option)
- [ ] Thread pool creation and management
- [ ] Distribute rosbags across threads
- [ ] Collect results from all threads
- [ ] Generate batch_summary.json
- [ ] Progress reporting (optional)

### Phase 6: Polish
- [ ] Comprehensive error handling
- [ ] Logging system integration
- [ ] README with usage examples
- [ ] Launch file for easy execution
- [ ] Integration testing with real dataset

---

## Key Technical Notes

### rosbag2 API Usage

**Opening a bag**:
```cpp
rosbag2_cpp::Reader reader;
reader.open(bag_path);
auto metadata = reader.get_metadata();
```

**Filtering topics**:
```cpp
rosbag2_storage::StorageFilter filter;
filter.topics.push_back("/vehicle/status/control_mode");
reader.set_filter(filter);
```

**Reading messages**:
```cpp
while (reader.has_next()) {
  auto bag_message = reader.read_next();
  auto timestamp = bag_message->time_stamp;
  auto topic = bag_message->topic_name;

  // Deserialize
  rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
  rclcpp::Serialization<ControlModeReport> serializer;
  ControlModeReport msg;
  serializer.deserialize_message(&serialized_msg, &msg);
}
```

**Writing a bag**:
```cpp
rosbag2_cpp::Writer writer;
rosbag2_storage::StorageOptions storage_options;
storage_options.uri = output_path;
storage_options.storage_id = "mcap";

rosbag2_cpp::ConverterOptions converter_options;
converter_options.output_serialization_format = "cdr";

writer.open(storage_options, converter_options);

// Create topic
rosbag2_storage::TopicMetadata topic_metadata;
topic_metadata.name = topic_name;
topic_metadata.type = type_name;
topic_metadata.serialization_format = "cdr";
writer.create_topic(topic_metadata);

// Write message
writer.write(bag_message);
```

### Performance Considerations

1. **Memory efficiency**: Process one rosbag at a time per thread, don't load entire bags into memory
2. **I/O optimization**: MCAP format is more efficient than SQLite3 for sequential writes
3. **Thread count**: Optimal = number of physical cores (not hyperthreads) due to I/O bound nature
4. **Disk space**: Check available space before processing (output size ≈ input size × num_overrides × topic_fraction)

---

## Testing Strategy

### Unit Tests
- OverrideDetector: Test state transition logic with mock messages
- BagSplitter: Test time/topic filtering with small test bags
- TimeRange merging: Test overlapping range consolidation

### Integration Tests
- Process sample rosbag with known override events
- Verify output MCAP files contain correct time ranges
- Verify only specified topics are preserved
- Test parallel processing with multiple bags

### Edge Cases
- Rosbag with no override events
- Override at very start/end of bag
- Multiple rapid overrides (overlapping margins)
- Missing ControlModeReport topic
- Corrupted rosbag files
- Very brief overrides (<0.1s)

---

## Future Enhancements

1. **Additional override indicators**: Support RTC status, diagnostic messages
2. **Interactive mode**: GUI for selecting specific overrides to extract
3. **Metadata extraction**: Extract vehicle state, trajectory info during overrides
4. **Comparative analysis**: Generate plots comparing manual vs autonomous behavior
5. **Cloud integration**: Direct upload to S3/cloud storage
6. **Real-time mode**: Monitor live rosbag and trigger extraction on override

---

## References

- rosbag2 documentation: https://github.com/ros2/rosbag2
- MCAP format spec: https://mcap.dev/
- Autoware vehicle messages: `autoware_vehicle_msgs/msg/ControlModeReport.msg`
