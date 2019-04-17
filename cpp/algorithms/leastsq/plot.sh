#!/bin/bash
d=res
if [ $# -ge 1 ]; then
  d=$1
fi
qplot -3d $d/sample.dat pt 7 ps 2 $d/target.dat w l $d/test.dat w l
