#!/bin/bash
usage="usage: x++ [options] FILES [-- OPTIONS GIVEN TO g++]
  options:
    -ww          : use extra warning options
    -oldlora     : use old-lora
    -lora        : use lora
    -hmrl        : use hmrl
    -slora       : use lora of the skyai
    -skyai       : use skyai
    -oct         : use liboctave
    -ode         : use ODE
    -gl          : use OpenGL
    -cv          : use OpenCV
    -eig         : using Eigen
    -sg          : using Shogun
    -make        : make libraries
    -opts 'OPTS' : OPTS is directly given to g++
    -help        : show this"
source ~/bin/lib/bash/option-parser.sh
set_single_opts help ww oldlora lora hmrl slora skyai oldngnet oct ode gl cv eig sg make
parse_opts "$@"

if [ `opt help 0` -eq 1 ];then
  echo "$usage"
  exit 1
fi

CXX=g++
CXXFLAGS="-g -Wall -rdynamic -O2 -march=i686"
if [ `opt ww 0` -eq 1 ];then
  echo 'more warning options' > /dev/stderr
  CXXFLAGS+=" -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Woverloaded-virtual -Winline"
fi
LIBS="-I/usr/include -I/usr/local/include"
LDLIBS="-lm -L/usr/local/lib"
##-------------------------------------------------------------------------------------------
## for libskyai
if [ `opt skyai 0` -eq 1 ]; then
  echo 'using libskyai' > /dev/stderr
  LIBSKYAI="$HOME/prg/skyai-dev2/build"
  LIBS+=" -I$LIBSKYAI/include/skyai"
  # LDLIBS+=" -L$LIBSKYAI/lib $LIBSKYAI/lib/*.o -lskyai"
  LDLIBS+=" -L$LIBSKYAI/lib/skyai -lskyai_mcore -lskyai_mstd -lskyai"
  LDLIBS+=" -Wl,-rpath $LIBSKYAI/lib/skyai"
  LDLIBS+=" -lboost_filesystem -lboost_regex"
fi
##-------------------------------------------------------------------------------------------
## for liblora in the skyai
if [ `opt slora 0` -eq 1 ]; then
  echo 'using liblora in the skyai' > /dev/stderr
  LIBSLORA="$HOME/prg/skyai-dev2/build"
  LIBS+=" -I$LIBSLORA/include/skyai"
  LDLIBS+=" -L$LIBSLORA/lib/skyai"
  if [ `opt oct 0` -eq 1 ]; then
    LDLIBS+=" -llora_oct"
  fi
  if [ `opt ode 0` -eq 1 ]; then
    LDLIBS+=" -llora_ode"
  fi
  LDLIBS+=" -llora_std"
  LDLIBS+=" -Wl,-rpath $LIBSLORA/lib/skyai"
fi
##-------------------------------------------------------------------------------------------
## for Octave
if [ `opt oct 0` -eq 1 ]; then
  echo 'using liboctave' > /dev/stderr
  LIBS+=" -I/usr/include/octave-`octave-config -v`"
  LDLIBS+=" -L/usr/lib/octave-`octave-config -v` -loctave -lcruft -Wl,-rpath /usr/lib/octave-`octave-config -v`"
  #LDLIBS+=" -ldl -lfftw3 -L/usr/lib/atlas -latlas -llapack -lblas -lg2c"
  LDLIBS+=" -ldl -lfftw3 -L/usr/lib/atlas -latlas -llapack -lblas"
fi
##-------------------------------------------------------------------------------------------
## for ODE
if [ `opt ode 0` -eq 1 ]; then
  echo 'using ODE' > /dev/stderr
  BASEPREFIX=$HOME/prg
  LIBS+=" -I$BASEPREFIX/libode/ode-0.10.1/include"
  LIBS+=" -DODE_MINOR_VERSION=10 -DdDOUBLE" # for ODE-0.10.1
  LDLIBS+=" $BASEPREFIX/libode/ode-0.10.1/ode/src/.libs/libode.a"
  LDLIBS+=" $BASEPREFIX/libode/ode-0.10.1/drawstuff/src/.libs/libdrawstuff.a"
fi
##-------------------------------------------------------------------------------------------
## for OpenGL
if [ `opt gl 0` -eq 1 ] || [ `opt ode 0` -eq 1 ]; then
  echo 'using OpenGL' > /dev/stderr
  LDLIBS+=" -lSM -lICE -lGL -L/usr/X11R6/lib -lXext -lX11 -ldl -lGLU -lpthread"
fi
##-------------------------------------------------------------------------------------------
## for OpenCV
if [ `opt cv 0` -eq 1 ]; then
  echo 'using OpenCV' > /dev/stderr
  LIBS+=" `pkg-config opencv --cflags`"
  LDLIBS+=" `pkg-config opencv --libs`"
fi
##-------------------------------------------------------------------------------------------
## for Eigen
if [ `opt eig 0` -eq 1 ]; then
  echo 'using Eigen' > /dev/stderr
  LIBS+=" -I/usr/include/eigen3"
fi
##-------------------------------------------------------------------------------------------
## for Shogun
if [ `opt sg 0` -eq 1 ]; then
  echo 'using Shogun' > /dev/stderr
  LDLIBS+=" -lshogun"
fi
##-------------------------------------------------------------------------------------------

options=`opt opts`

if [ `opt make 0` -eq 1 ]; then
  echo 'make libs...' > /dev/stderr
  if [ -n "${LIBHMRL:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBHMRL/src" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBHMRL/src; then exit 1; fi
  fi
  if [ -n "${LIBSKYAI:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBSKYAI/src" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBSKYAI/src; then exit 1; fi
  fi
  if [ -n "${LIBNGNET:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBNGNET" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBNGNET; then exit 1; fi
  fi
  if [ -n "${LIBOLDLORA:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBOLDLORA/src" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBOLDLORA/src; then exit 1; fi
  fi
  if [ -n "${LIBLORA:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBLORA/src" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBLORA/src; then exit 1; fi
  fi
  if [ -n "${LIBSLORA:-}" ];then
    echo "make ${fargs[@]} $options -C $LIBSLORA/src" > /dev/stderr
    if ! make ${fargs[@]} $options -C $LIBSLORA/src; then exit 1; fi
  fi
fi

echo "g++ $CXXFLAGS ${fargs[@]} $options $LIBS $LDLIBS" > /dev/stderr
g++ $CXXFLAGS ${fargs[@]} $options $LIBS $LDLIBS

