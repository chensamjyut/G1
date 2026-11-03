cmake_minimum_required(VERSION 3.5)
project(local_planner)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(serial REQUIRED)

add_executable(localPlanner src/localPlanner.cpp)
add_executable(inspectionPathFollower src/inspectionPathFollower.cpp)
# the upstream project but no pcl_ros header or symbol is referenced.  Keeping
# it here makes a minimal Humble installation fail configuration needlessly.
ament_target_dependencies(localPlanner rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs message_filters pcl_conversions)
ament_target_dependencies(inspectionPathFollower rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs message_filters pcl_conversions serial)

# Optional explicit workstation linker override; no assumed host path.
set(G1_CXX_RUNTIME "" CACHE FILEPATH "Compatible libstdc++.so.6 override, normally empty")
if(G1_CXX_RUNTIME)
  if(NOT EXISTS "${G1_CXX_RUNTIME}")
    message(FATAL_ERROR "G1_CXX_RUNTIME file does not exist")
  endif()
  target_link_libraries(localPlanner "${G1_CXX_RUNTIME}")
  target_link_libraries(inspectionPathFollower "${G1_CXX_RUNTIME}")
endif()
install(TARGETS
  localPlanner
  inspectionPathFollower
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY
  config
  launch
  paths
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
