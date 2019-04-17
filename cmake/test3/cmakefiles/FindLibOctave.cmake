# set(LIBOCTAVE_LIB_DESCR "Path to the directory of OpenNI libraries" CACHE INTERNAL "Description" )
# set(LIBOCTAVE_INCLUDE_DESCR "Path to the directory of OpenNI includes" CACHE INTERNAL "Description" )

find_path(LIBOCTAVE_octave_INCLUDE_DIR NAMES octave.h PATH_SUFFIXES octave PATHS /lib/include /lib/local/include)
if(NOT LIBOCTAVE_octave_INCLUDE_DIR)
  set(LIBOCTAVE_octave_INCLUDE_DIR "LIBOCTAVE_octave_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path to the directory that contains octave.h" FORCE)
endif()

find_library(LIBOCTAVE_octave_LIBRARY NAMES octave PATHS /usr/lib /usr/local/lib)
find_library(LIBOCTAVE_dl_LIBRARY     NAMES dl     PATHS /usr/lib /usr/local/lib)
find_library(LIBOCTAVE_fttw3_LIBRARY  NAMES fttw3  PATHS /usr/lib /usr/local/lib)
if(NOT LIBOCTAVE_fttw3_LIBRARY)
  set(LIBOCTAVE_fttw3_LIBRARY "")
endif()
find_library(LIBOCTAVE_atlas_LIBRARY  NAMES atlas  PATHS /usr/lib /usr/local/lib)
find_library(LIBOCTAVE_lapack_LIBRARY NAMES lapack PATHS /usr/lib /usr/local/lib)
find_library(LIBOCTAVE_blas_LIBRARY   NAMES blas   PATHS /usr/lib /usr/local/lib)

set (LIBOCTAVE_FOUND "NO")
if(LIBOCTAVE_octave_INCLUDE_DIR AND LIBOCTAVE_octave_LIBRARY)
  set (LIBOCTAVE_FOUND "YES")
endif()


set (LIBOCTAVE_INCLUDE_DIRS ${LIBOCTAVE_octave_INCLUDE_DIR}/..)
set (LIBOCTAVE_LIBRARIES
    ${LIBOCTAVE_octave_LIBRARY}
    ${LIBOCTAVE_dl_LIBRARY}
    ${LIBOCTAVE_fttw3_LIBRARY}
    ${LIBOCTAVE_atlas_LIBRARY}
    ${LIBOCTAVE_lapack_LIBRARY}
    ${LIBOCTAVE_blas_LIBRARY}
  )


