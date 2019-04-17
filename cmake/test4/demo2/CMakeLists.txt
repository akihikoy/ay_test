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
add_executable (demo2.out demo2.cpp)
target_link_libraries (demo2.out
    ${X_LIBRARIES}
  )
