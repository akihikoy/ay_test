cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# NOTE: we assume that ODE is built in $HOME/prg/libode/ode-latest/ from source.

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

rosbuild_add_executable(joint_chain1_node
      src/joint_chain1_node.cpp
      src/joint_chain1.cpp)
add_definitions(-DdDOUBLE)
include_directories($ENV{HOME}/prg/libode/ode-latest/include)
target_link_libraries(joint_chain1_node
      $ENV{HOME}/prg/libode/ode-latest/ode/src/.libs/libode.a
      $ENV{HOME}/prg/libode/ode-latest/drawstuff/src/.libs/libdrawstuff.a
      SM ICE GL Xext X11 GLU)

rosbuild_add_executable(arm7_door_push_node
      src/arm7_door_push_node.cpp
      src/arm7_door_push.cpp)
add_definitions(-DdDOUBLE)
include_directories($ENV{HOME}/prg/libode/ode-latest/include)
target_link_libraries(arm7_door_push_node
      $ENV{HOME}/prg/libode/ode-latest/ode/src/.libs/libode.a
      $ENV{HOME}/prg/libode/ode-latest/drawstuff/src/.libs/libdrawstuff.a
      SM ICE GL Xext X11 GLU)

rosbuild_add_executable(gripper_node
      src/gripper_node.cpp
      src/gripper.cpp)
add_definitions(-DdDOUBLE)
include_directories($ENV{HOME}/prg/libode/ode-latest/include /usr/include/eigen3)
target_link_libraries(gripper_node
      $ENV{HOME}/prg/libode/ode-latest/ode/src/.libs/libode.a
      $ENV{HOME}/prg/libode/ode-latest/drawstuff/src/.libs/libdrawstuff.a
      SM ICE GL Xext X11 GLU)

