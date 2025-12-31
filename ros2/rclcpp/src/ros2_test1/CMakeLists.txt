cmake_minimum_required(VERSION 3.8)
project(ros2_test1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ros2_test1_msgs REQUIRED)

include_directories(include)

add_library(ros2_test1_components SHARED
  src/cpp_talker_node.cpp
  src/cpp_listener_node.cpp
  src/cpp_srv_client.cpp
  src/cpp_action_client.cpp
)

ament_target_dependencies(ros2_test1_components
  rclcpp
  rclcpp_action
  rclcpp_components
  ros2_test1_msgs
)

# register components
# rclcpp_components_register_nodes(ros2_test1_components
#   "ros2_test1::TalkerNode"
#   "ros2_test1::ListenerNode"
#   "ros2_test1::SrvClientNode"
#   "ros2_test1::ActionClientNode"
# )

# build executables
add_executable(cpp_talker_exec src/cpp_talker_main.cpp)
target_link_libraries(cpp_talker_exec ros2_test1_components)
ament_target_dependencies(cpp_talker_exec rclcpp rclcpp_action)

add_executable(cpp_listener_exec src/cpp_listener_main.cpp)
target_link_libraries(cpp_listener_exec ros2_test1_components)
ament_target_dependencies(cpp_listener_exec rclcpp rclcpp_action)

add_executable(cpp_srvc_exec src/cpp_srvc_main.cpp)
target_link_libraries(cpp_srvc_exec ros2_test1_components)
ament_target_dependencies(cpp_srvc_exec rclcpp rclcpp_action)

add_executable(cpp_actionc_exec src/cpp_actionc_main.cpp)
target_link_libraries(cpp_actionc_exec ros2_test1_components)
ament_target_dependencies(cpp_actionc_exec rclcpp rclcpp_action)

# install
install(TARGETS
  ros2_test1_components
  cpp_talker_exec
  cpp_listener_exec
  cpp_srvc_exec
  cpp_actionc_exec
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
