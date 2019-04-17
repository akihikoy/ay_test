cmake_minimum_required (VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project (TEST4)


include(../test3/FindOpenCV.cmake)
if(NOT OPENCV_FOUND)
  message(WARNING, " OpenCV library is not found." )
endif()

find_package(Boost COMPONENTS
  filesystem
  REQUIRED)
if(NOT Boost_FOUND)
  message(WARNING, " Boost library is not found." )
endif()
if(NOT Boost_FILESYSTEM_FOUND)
  message(WARNING, " Boost-filesystem library is not found." )
endif()

function(lib_configs)
  # message("arg1=${arg1} : ARGN=${ARGN}")
  foreach(a ${ARGN})
    message("a=${a}")
    if(${a} STREQUAL boost)
      message("using boost")
      set(L_INCLUDE_DIRS  ${L_INCLUDE_DIRS}  ${Boost_INCLUDE_DIRS})
      set(L_LIBRARY_DIRS  ${L_LIBRARY_DIRS}  ${Boost_LIBRARY_DIRS})
      set(L_LIBRARIES     ${L_LIBRARIES}     ${Boost_LIBRARIES})
    elseif(${a} STREQUAL cv)
      message("using cv")
      set(L_INCLUDE_DIRS  ${L_INCLUDE_DIRS}  ${OPENCV_INCLUDE_DIRS})
      set(L_LIBRARY_DIRS  ${L_LIBRARY_DIRS}  ${OPENCV_LIBRARY_DIRS})
      set(L_LIBRARIES     ${L_LIBRARIES}     ${OPENCV_LIBRARIES})
    endif()
  endforeach()
  set(X_INCLUDE_DIRS  ${L_INCLUDE_DIRS} PARENT_SCOPE)
  set(X_LIBRARY_DIRS  ${L_LIBRARY_DIRS} PARENT_SCOPE)
  set(X_LIBRARIES     ${L_LIBRARIES}    PARENT_SCOPE)
endfunction()

# message("f='${f}' arg1='${arg1}' arg2='${arg2}' ARGV2='${ARGV2}'")


# add_subdirectory(demo1)
# add_subdirectory(demo2)


# # # # # #
lib_configs(boost)
message("X_INCLUDE_DIRS= ${X_INCLUDE_DIRS}")
message("X_LIBRARY_DIRS= ${X_LIBRARY_DIRS}")
message("X_LIBRARIES= ${X_LIBRARIES}")

include_directories(
    ${X_INCLUDE_DIRS}
  )
link_directories (
    ${X_LIBRARY_DIRS}
  )
add_executable (demo1.out demo1/demo1.cpp)
target_link_libraries (demo1.out
    ${X_LIBRARIES}
  )

# # # # # #
lib_configs(cv)
message("X_INCLUDE_DIRS= ${X_INCLUDE_DIRS}")
message("X_LIBRARY_DIRS= ${X_LIBRARY_DIRS}")
message("X_LIBRARIES= ${X_LIBRARIES}")

include_directories(
    ${X_INCLUDE_DIRS}
  )
link_directories (
    ${X_LIBRARY_DIRS}
  )
add_executable (demo2.out demo2/demo2.cpp)
target_link_libraries (demo2.out
    ${X_LIBRARIES}
  )

# # # # # #

