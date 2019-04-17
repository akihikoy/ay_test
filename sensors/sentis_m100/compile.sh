#!/bin/bash
out=${1/.cpp/.out}
cmd="g++ -g -Wall -rdynamic -O2 $@ -o $out -I/usr/include/libm100 -lm100 -lrt -lGLU -lGL -lSM -lICE -lX11 -lXext -lglut -lXmu -lXi"
if type becho >& /dev/null;then echo=becho;else echo=echo;fi
$echo $cmd
if $cmd;then
  $echo "Generated: $out"
fi
