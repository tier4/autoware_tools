cmake_minimum_required(VERSION 3.14)
project(autoware_mppi_evaluator)

find_package(autoware_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)
autoware_package()

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR streamlit_explorer
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/mppi_batch_evaluator.cpp
)

pybind11_add_module(mppi_optimizer_py MODULE
  src/mppi_pybind.cpp
)
target_link_libraries(mppi_optimizer_py PRIVATE ${PROJECT_NAME})

install(TARGETS mppi_optimizer_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/mppi_evaluate_bag.py
  scripts/mppi_streamlit_explorer.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_auto_add_gtest(test_mppi_metrics
    test/test_mppi_metrics.cpp
  )
  target_link_libraries(test_mppi_metrics ${PROJECT_NAME})

  add_test(
    NAME test_mcap_reader
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_mcap_reader.py
  )
  set_tests_properties(test_mcap_reader PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/ament_cmake_python/${PROJECT_NAME}"
  )

  add_test(
    NAME test_dataset_io
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_dataset_io.py
  )
  set_tests_properties(test_dataset_io PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/ament_cmake_python/${PROJECT_NAME}"
  )
endif()

ament_auto_package(INSTALL_TO_SHARE config)
