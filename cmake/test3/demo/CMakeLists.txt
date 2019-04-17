include_directories(${FOO_INCLUDE_DIRS})
link_directories (${FOO_LIBRARY_DIRS})

add_executable (demo1.out demo1.cpp)
target_link_libraries (demo1.out ${FOO_LIBRARIES})

#-------

include_directories(${BAR_INCLUDE_DIRS})
link_directories (${BAR_LIBRARY_DIRS})

# link_directories (${LIBOCTAVE_LINK_DIRS})

add_executable (demo2.out demo2.cpp)
target_link_libraries (demo2.out ${BAR_LIBRARIES} ${LIBOCTAVE_LIBRARIES})

#-------

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories (${OPENCV_LIBRARY_DIRS})

add_executable (demo3.out demo3.cpp)
target_link_libraries (demo3.out ${BAR_LIBRARIES} ${LIBOCTAVE_LIBRARIES} ${OPENCV_LIBRARIES})

#-------

include_directories(${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

# link_directories (${LIBOCTAVE_LINK_DIRS})

add_executable (demo4.out demo4.cpp)
target_link_libraries (demo4.out ${BAR_LIBRARIES} ${LIBOCTAVE_LIBRARIES} ${Boost_LIBRARIES})

#-------

include_directories(${ODE_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})

add_executable (odedemo1.out odedemo1.cpp)
target_link_libraries (odedemo1.out ${ODE_LIBRARIES} ${OPENGL_LIBRARIES})

execute_process (COMMAND
    ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/textures
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
