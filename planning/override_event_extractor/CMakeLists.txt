cmake_minimum_required(VERSION 3.14)
project(override_event_extractor)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_executable(${PROJECT_NAME}_node
  src/main.cpp
  src/override_detector.cpp
  src/bag_splitter.cpp
  src/bag_processor.cpp
  src/parallel_executor.cpp
)

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
