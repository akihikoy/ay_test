#!/bin/bash
#\file    sync.sh
#\brief   Sync files from original (private use).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.06, 2017

base=~/ros_ws/ay_tools/ay_py/src/ay_py/core/
files=(
_rostf.py
geom.py
util.py
traj.py
)
files2=""
for ((i=0; i<$((${#files[@]})); i++)) ; do
  files2="$files2 ${base}/./${files[i]}"
done
# echo $files2
rsync -azv -R -L ${files2} .

