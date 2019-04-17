# The name of our project is "HELLO". CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


project (TEST3)

set (WITH_LIBOCTAVE ON CACHE BOOL "Include LibOctave support")
set (WITH_BOOST ON CACHE BOOL "Include Boost support")
set (WITH_OPENCV ON CACHE BOOL "Include OpenCV support")
set (WITH_ODE ON CACHE BOOL "Include ODE support")

set (FOO_INCLUDE_DIRS ${TEST3_SOURCE_DIR}/foo/include)
set (FOO_LIBRARY_DIRS ${TEST3_BINARY_DIR}/foo)
set (FOO_LIBRARIES foo)

set (BAR_INCLUDE_DIRS ${TEST3_SOURCE_DIR}/bar/include)
set (BAR_LIBRARY_DIRS ${FOO_LIBRARY_DIRS} ${TEST3_BINARY_DIR}/bar)
set (BAR_LIBRARIES bar foo)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmakefiles)

if(WITH_LIBOCTAVE)
  include(FindLibOctave)  # NOTE: don't append .cmake
  if(NOT LIBOCTAVE_FOUND)
    message(WARNING, " Octave library is not found or does not have Octave library." )
  endif()
endif()

if(WITH_BOOST)
  find_package(Boost COMPONENTS
    filesystem
    REQUIRED)
  if(NOT Boost_FOUND)
    message(WARNING, " Boost library is not found." )
  endif()
  if(NOT Boost_FILESYSTEM_FOUND)
    message(WARNING, " Boost-filesystem library is not found." )
  endif()
endif()

if(WITH_OPENCV)
  include(FindOpenCV)  # NOTE: don't append .cmake
  if(NOT OPENCV_FOUND)
    message(WARNING, " OpenCV library is not found or does not have OpenCV library." )
  endif()
endif()

if(WITH_ODE)
  include(FindODE)  # NOTE: don't append .cmake
  if(NOT ODE_FOUND)
    message(WARNING, " ODE library is not found." )
  endif()
  add_definitions(-DODE_MINOR_VERSION=10 -DdDOUBLE)
  find_package(OpenGL COMPONENTS
    filesystem
    REQUIRED)
  if(NOT (OPENGL_FOUND AND OPENGL_GLU_FOUND))
    message(WARNING, " ODE library requires OpenGL and GLU, but not found." )
  endif()
endif()

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
add_subdirectory (foo)
add_subdirectory (bar)
add_subdirectory (demo)


# Create a library called "Hello" which includes the source file "hello.cxx".
# The extension is already found. Any number of sources could be listed here.
# add_library (test test.cpp)

# Make sure the linker can find the Hello library once it is built.
# make_directory(${HELLO_BINARY_DIR}/lib)
# link_directories (${TEST3_BINARY_DIR})

# Add executable called "helloDemo" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
# add_executable (demo1.out demo/demo.cpp)
# Link the executable to the Hello library.
# target_link_libraries (demo1.out test)


