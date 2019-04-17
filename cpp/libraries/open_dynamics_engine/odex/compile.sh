#!/bin/bash
usage="usage: ./compile.sh [options] FILE [-- ADDITIONAL OPTIONS FOR g++]
  options:
    -eig         : using Eigen
    -sg          : using Shogun
    -ode         : using ODE
    -help        : show this"
CXXFLAGS="-g -Wall -O3"
LIBS_EIG="-I/usr/include/eigen3"
LDLIBS_SHOGUN="-lshogun"
ODE_PATH=/home/akihiko/prg/libode/ode-0.13
LIBS_ODE="-I$ODE_PATH/include -DdDOUBLE"
LDLIBS_ODE="-lm $ODE_PATH/ode/src/.libs/libode.a $ODE_PATH/drawstuff/src/.libs/libdrawstuff.a -lSM -lICE -lGL -L/usr/X11R6/lib -lXext -lX11 -ldl -lGLU -lpthread"

TARGET=
LIBS=
LDLIBS=
while true; do
  case "$1" in
    -help|--help) echo "usage: $usage"; exit 0 ;;
    -eig) LIBS="$LIBS $LIBS_EIG"; shift 1 ;;
    -sg)  LDLIBS="$LDLIBS $LDLIBS_SHOGUN"; shift 1 ;;
    -ode) LIBS="$LIBS $LIBS_ODE"; LDLIBS="$LDLIBS $LDLIBS_ODE"; shift 1 ;;
    '') break ;;
    --) shift 1; break ;;
    *)
      if [ -n "$TARGET" ];then echo "usage: $usage"; exit 0; fi
      if [ "$1" != "${1/.cpp/.out}" ];then TARGET="$1 -o ${1/.cpp/.out}"
      else TARGET="$1"; fi
      shift 1 ;;
  esac
done

echo g++ $CXXFLAGS $TARGET $@ $LIBS $LDLIBS
g++ $CXXFLAGS $TARGET $@ $LIBS $LDLIBS
