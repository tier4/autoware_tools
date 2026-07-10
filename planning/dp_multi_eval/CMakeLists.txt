cmake_minimum_required(VERSION 3.5)
project(dp_multi_eval)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_python_install_package(${PROJECT_NAME})

ament_auto_package(
  INSTALL_TO_SHARE
    config
)

install(PROGRAMS
  scripts/run_single_job.py
  scripts/compute_metrics.py
  scripts/render_video.py
  scripts/generate_manifest.py
  scripts/run_orchestrator.py
  scripts/build_dashboard.py
  scripts/run_all.py
  scripts/app.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})
