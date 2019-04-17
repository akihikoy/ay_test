#!/bin/bash

datafile=res/kernel.dat

# N=$(($(head -1 $datafile | wc -w)-1))
N=$(head -1 $datafile | wc -w)

pline=""
for ((i=2; i<=$N; i+=1)); do
  pline+=" res/kernel.dat u 1:$i w l"
done

pline+=" res/f_trg.dat res/f.dat w l"

qplot $pline
