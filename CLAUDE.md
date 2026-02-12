# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **autoware_tools**, a collection of 41 development and analysis tools for Autoware. These packages are **not needed at runtime** - they're for benchmarking, debugging, tuning, calibrating, and analyzing the autonomous driving system.

**Repository:** https://github.com/tier4/autoware_tools.git
**ROS Distro:** Humble
**Default Branch:** main

## Directory Structure

Tools are organized by Autoware component:

```
autoware_tools/
├── common/                    # 15 packages - RVIZ plugins and debug utilities
├── planning/                  # 6 packages - Planning analysis and debugging
├── control/                   # 2 packages - Control analysis and debugging
├── localization/              # 2 packages - Localization evaluation
├── map/                       # 6 packages - Lanelet2 and pointcloud utilities
├── vehicle/                   # 5 packages - Vehicle parameter estimation
├── system/                    # 1 package - Diagnostic monitoring
├── evaluation/                # 1 package - Metrics visualization
├── simulator/                 # 1 package - Simulator compatibility tests
├── driving_environment_analyzer/
├── control_data_collecting_tool/
├── bag2lanelet/
└── autoware_dependency_checker/
```

### Key Packages by Category

**RVIZ Plugins (common/):**
- `mission_planner_rviz_plugin` - Interactive mission planning interface
- `rtc_manager_rviz_plugin` - Runtime configuration manager
- `tier4_control_rviz_plugin` - Control visualization and debugging
- `tier4_calibration_rviz_plugin` - Sensor calibration interface
- `tier4_automatic_goal_rviz_plugin` - Automatic goal setting

**Planning Tools (planning/):**
- `override_event_extractor` - Extract manual override segments from rosbags
- `autoware_static_centerline_generator` - Generate static centerlines for maps
- `autoware_rtc_replayer` - Replay RTC (Run Time Coordinator) events
- `autoware_planning_data_analyzer` - Analyze planning behavior from logs
- `autoware_route_client` - Command-line route planning client

**Map Tools (map/):**
- `autoware_lanelet2_map_utils` - Lanelet2 map manipulation utilities
- `autoware_lanelet2_map_validator` - Validate lanelet2 maps
- `autoware_pointcloud_divider` / `autoware_pointcloud_merger` - Split/merge pointclouds
- `autoware_tp_manager` - Traffic participant manager

**Control Tools (control/):**
- `vehicle_cmd_analyzer` - Analyze vehicle command characteristics
- `control_debug_tools` - Control debugging utilities

**Vehicle Tools (vehicle/):**
- `parameter_estimator` - Estimate vehicle parameters from data
- `time_delay_estimator` - Estimate actuation time delays
- `pitch_checker` - Validate vehicle pitch estimation

## Build System & Commands

### Setup Dependencies

```bash
# Install vcstool if not available
sudo apt install python3-vcstool

# Import build dependencies (from repository root)
vcs import < build_depends.repos
rosdep update
rosdep install --from-paths . --ignore-src -y
```

### Building

```bash
# Build all tools packages
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build specific package
colcon build --packages-select <package_name> --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build package and its dependencies
colcon build --packages-up-to <package_name> --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source after building
source install/setup.bash
```

**Important:** Always use `--symlink-install` to avoid copying Python files. Always build with `-DCMAKE_BUILD_TYPE=Release` unless debugging.

### Testing

```bash
# Test all packages
colcon test

# Test specific package with verbose output
colcon test --packages-select <package_name> --event-handlers console_direct+ --return-code-on-test-failure

# Run C++ test executable directly (after building)
./install/<package_name>/lib/<package_name>/test_<test_name>

# View test results
colcon test-result --verbose
```

### Linting

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run all linting checks
pre-commit run --all-files

# Run specific linters
pre-commit run clang-format --all-files
pre-commit run cpplint --all-files
pre-commit run flake8-ros --all-files
pre-commit run black --all-files

# Lint only modified files
pre-commit run --files <file1> <file2>
```

## Package Structure Pattern

All packages follow standard ROS 2 structure:

```
package_name/
├── package.xml              # ROS 2 metadata (dependencies, maintainer, license)
├── CMakeLists.txt           # Build configuration using autoware_cmake
├── src/                     # C++ source files
├── include/<package_name>/  # Public headers
├── config/                  # YAML parameter files
├── launch/                  # ROS launch files (Python or XML)
├── test/                    # Unit tests (gtest, pytest)
└── README.md                # Package-specific documentation
```

### CMakeLists.txt Pattern

```cmake
cmake_minimum_required(VERSION 3.14)
project(package_name)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Auto-detect and add library/executable
ament_auto_add_library(lib_name src/file1.cpp src/file2.cpp)
ament_auto_add_executable(exe_name src/main.cpp)

# Tests
if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(test_name test/test_file.cpp)
  target_link_libraries(test_name lib_name)
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
```

### package.xml Pattern

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>package_name</name>
  <version>0.1.0</version>
  <description>Package description</description>
  <maintainer email="dev@tier4.jp">Tier4</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake_auto</buildtool_depend>
  <buildtool_depend>autoware_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>dependency_package</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>autoware_lint_common</test_depend>

  <export><build_type>ament_cmake</build_type></export>
</package>
```

## Code Style & Conventions

### C++ Style

- **Base Style:** Google C++ Style Guide
- **Line Limit:** 100 characters
- **Formatting:** Enforced by `.clang-format` (clang-format v19)
- **Linting:** cpplint with custom filters (see `CPPLINT.cfg`)
- **Pointer Alignment:** Middle (`int * ptr`)
- **Braces:** Custom wrapping (after class, function, namespace, struct)
- **Include Order:**
  1. Local package headers (`"header.hpp"`)
  2. Other package headers (`<package/header.hpp>`)
  3. Message/service headers (`*_msgs/*`, `*_srvs/*`)
  4. Boost headers (`boost/*`)
  5. C system headers (`<stdio.h>`)
  6. C++ system headers (`<vector>`)

### Python Style

- **Line Limit:** 100 characters
- **Formatter:** Black (enforced by pre-commit)
- **Linter:** flake8-ros (ROS-specific rules)
- **Import Sorting:** isort with black profile
- **Configuration:** See `setup.cfg`

### Namespace Convention

Use nested namespaces matching package structure:
```cpp
namespace autoware::tools::package_name
{
// Code here
}  // namespace autoware::tools::package_name
```

## Development Workflow

### Creating a New Package

1. Choose appropriate subdirectory (common/, planning/, control/, etc.)
2. Create package skeleton:
   ```bash
   ros2 pkg create <package_name> \
     --build-type ament_cmake \
     --dependencies rclcpp
   ```
3. Add `autoware_cmake` to build dependencies
4. Update `CMakeLists.txt` to use `autoware_package()` macro
5. Implement functionality in `src/` and `include/`
6. Add configuration files in `config/`
7. Add tests in `test/`
8. Write package README.md explaining usage

### Modifying Existing Code

1. **Read the package README first** - most packages have detailed documentation
2. Check existing patterns in similar packages
3. Verify dependencies in `build_depends.repos`
4. Run linting before committing: `pre-commit run --files <modified_files>`
5. Ensure tests pass: `colcon test --packages-select <package>`

### Example: Running a Tool

Many tools are ROS 2 executables that process rosbags or interact with running systems:

```bash
# Source the workspace
source install/setup.bash

# Run a tool node
ros2 run <package_name> <executable_name> [args]

# Example: Extract override events from rosbags
ros2 run override_event_extractor override_event_extractor_node \
  --input-dir /data/rosbags \
  --output-dir /data/override_segments \
  --recursive
```

## Dependencies

External dependencies are managed in `build_depends.repos`:

- `autoware_cmake` - CMake utilities
- `autoware.core` - Core Autoware packages
- `autoware_msgs` - Message definitions
- `autoware_utils` - Common utilities
- `autoware_lanelet2_extension` - Lanelet2 map extensions
- Various vendor packages (grid_map, graph_tool_msgs, etc.)

Update dependencies:
```bash
vcs import < build_depends.repos
vcs pull
```

## Testing Strategy

### Unit Tests

- **C++:** gtest framework (`ament_add_ros_isolated_gtest`)
- **Python:** pytest framework
- Place tests in `test/` directory
- Name test files `test_*.cpp` or `test_*.py`

### Launch Tests

For testing ROS 2 launch files:
```python
# test/test_launch.py
import launch_testing
import pytest

# Test implementation
```

Run with:
```bash
colcon test --packages-select <package> --pytest-args -v
```

## Git Workflow

### Branch Strategy

Follow standard GitHub flow:
- `main` - Stable production branch
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches

### Creating Pull Requests

1. Fork or create a feature branch from `main`
2. Make changes and commit with semantic commit messages:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `refactor:` - Code refactoring
   - `docs:` - Documentation changes
   - `test:` - Test additions/modifications
   - `ci:` - CI/CD changes
3. Run linting: `pre-commit run --all-files`
4. Ensure tests pass: `colcon test`
5. Create PR to `main` with descriptive title and description
6. Address review comments

### Semantic PR Titles

PR titles must follow Conventional Commits format (enforced by CI):
```
<type>(<scope>): <description>

Examples:
feat(planning): add new trajectory analyzer tool
fix(control): correct vehicle command timestamp handling
docs(map): update lanelet2 validator README
```

## Common Pitfalls

1. **Not sourcing environment:** Always `source install/setup.bash` after building
2. **Missing dependencies:** Run `vcs import < build_depends.repos` before building new packages
3. **Forgetting symlink-install:** Python changes won't be reflected without `--symlink-install`
4. **Wrong build type:** Release mode is much faster than Debug for analysis tools
5. **Skipping pre-commit:** Failing CI checks waste time - run `pre-commit` locally first
6. **Not reading package READMEs:** Most packages have detailed usage examples in their README.md

## RVIZ Plugin Development

When developing RVIZ plugins (packages in `common/`):

1. Inherit from `rviz_common::Panel` or `rviz_common::Display`
2. Export plugin in `package.xml`:
   ```xml
   <export>
     <rviz_common plugin="${prefix}/plugins/plugin_description.xml"/>
   </export>
   ```
3. Create plugin description XML in `plugins/` directory
4. Use Qt for GUI components (Qt5 for ROS 2 Humble)
5. Register plugin class with `PLUGINLIB_EXPORT_CLASS` macro

## Rosbag Processing Tools

Several packages process rosbag files (override_event_extractor, planning_data_analyzer, etc.):

- Use `rosbag2_cpp` API for reading bags
- Support both db3 (SQLite) and mcap formats
- Implement parallel processing for batch analysis
- Provide progress feedback for long-running operations
- Output summary JSON files for results
- Filter topics to reduce output size

## Configuration Files

### .clang-format

Enforces code formatting (Google style, 100 char limit). Applied by pre-commit hooks.

### CPPLINT.cfg

Configures cpplint static analysis with custom filters for Autoware conventions.

### setup.cfg

Configures Python tools (flake8, isort, black) for 100-char line limit and black compatibility.

### build_depends.repos

VCS file listing external repository dependencies. Update with `vcs import/pull`.

## CI/CD

GitHub Actions workflows (`.github/workflows/`) provide:

- **build-and-test.yaml** - Build and test all packages on push/schedule
- **pre-commit.yaml** - Run linting checks on PRs
- **clang-tidy-pr-comments.yaml** - Static analysis with inline PR comments
- **semantic-pull-request.yaml** - Enforce semantic PR titles
- **build-and-test-differential.yaml** - Test only changed packages

CI uses Autoware Foundation's official Docker images with pre-installed dependencies.
