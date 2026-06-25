cmake_minimum_required(VERSION 3.5)
project(diffusion_planner_batch_eval)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_package(
  INSTALL_TO_SHARE
    config
)

install(PROGRAMS
  scripts/batch_diffusion_eval.py
  scripts/route_setup.py
  scripts/rosbag_utils.py
  scripts/trajectory_logger.py
  scripts/compare_trajectories.py
  scripts/analyze_model_comparison.py
  scripts/lanelet_boundary_check.py
  scripts/goal_stop_error.py
  scripts/npc_collision_check.py
  scripts/comfort_metrics.py
  scripts/compare_comfort.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})
