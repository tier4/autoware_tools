cmake_minimum_required(VERSION 3.5)
project(control_debug_tools)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_package(
  INSTALL_TO_SHARE
    scripts
)

install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/consistency_checker.py
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lateral_deviation_monitor.py
  DESTINATION share/${PROJECT_NAME}/scripts)

