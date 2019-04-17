#!/bin/bash -x

for i in `seq -w 0 2944`;do
  setting='set nokey'
  setting="$setting; set size ratio 1.0"
  setting="$setting; set xrange [-1.2:1.2];set yrange [-1.2:1.2]"
  setting="$setting; set xrange [-1.2:1.2];set yrange [-1.2:1.2]"
  setting="$setting; set terminal jpeg size 300 300"
  setting="$setting; set output 'frame$i.jpg'"
  qplot -s "$setting" frame$i.dat w p pt 7 ps 2
done

