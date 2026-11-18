cmake_minimum_required(VERSION 3.5)
project(autoware_perfect_tracker)

find_package(ament_cmake_auto REQUIRED)
find_package(autoware_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(autoware_vehicle_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tier4_api_utils REQUIRED)
find_package(tier4_external_api_msgs REQUIRED)
find_package(autoware_utils_geometry REQUIRED)
find_package(autoware_lanelet2_extension REQUIRED)
find_package(autoware_lanelet2_utils REQUIRED)
find_package(autoware_map_msgs REQUIRED)
find_package(autoware_motion_utils REQUIRED)
find_package(lanelet2_core REQUIRED)
find_package(lanelet2_routing REQUIRED)

autoware_package()

# 1. Create the Component Library
add_library(${PROJECT_NAME}_node SHARED
  src/perfect_tracker_node.cpp
)

target_include_directories(${PROJECT_NAME}_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  rclcpp_components
  autoware_planning_msgs
  autoware_vehicle_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_msgs
  tier4_api_utils
  tier4_external_api_msgs
  autoware_utils_geometry
  autoware_lanelet2_extension
  autoware_lanelet2_utils
  autoware_map_msgs
  autoware_motion_utils
  lanelet2_core
  lanelet2_routing
)

# 2. Generate the Executable Node
rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "autoware::simulator::perfect_tracker::PerfectTrackerNode"
  EXECUTABLE perfect_tracker_node
)

# 3. Install the Component Library
install(TARGETS ${PROJECT_NAME}_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 4. Install the Executable
install(TARGETS perfect_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
