set(foo_srcs
    src/foo1.cpp
    src/foo2.cpp
  )
add_library (foo SHARED ${foo_srcs})

include_directories (include)
