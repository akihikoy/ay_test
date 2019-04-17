set(bar_srcs
    src/bar1.cpp
    src/bar2.cpp
  )
add_library (bar SHARED ${bar_srcs})

include_directories (include)
include_directories (${FOO_INCLUDE_DIRS})
include_directories (${LIBOCTAVE_INCLUDE_DIRS})
