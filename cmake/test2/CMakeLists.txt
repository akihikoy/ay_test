# The name of our project is "HELLO". CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.4)
project (test2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
# add_subdirectory (Hello)
# add_subdirectory (Demo)

# Create a library called "Hello" which includes the source file "hello.cxx".
# The extension is already found. Any number of sources could be listed here.
# add_library (test test.cpp)
set(test_srcs
    src/test.cpp
    src/test2.cpp
  )
add_library (test SHARED ${test_srcs})

# Make sure the compiler can find include files from our Hello library.
include_directories (${test2_SOURCE_DIR}/include)

# Make sure the linker can find the Hello library once it is built.
# make_directory(${HELLO_BINARY_DIR}/lib)
link_directories (${test2_BINARY_DIR})

# Add executable called "helloDemo" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
add_executable (demo1.out demo/demo.cpp)
# Link the executable to the Hello library.
target_link_libraries (demo1.out test)


add_executable (demo2.out demo/demo2.cpp)
target_link_libraries (demo2.out test)

