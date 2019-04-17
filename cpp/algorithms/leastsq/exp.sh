#!/bin/bash

# qplot -s 'set logscale x' -cs 'u 1:2 ps 2' res/res-{LS,SVD}-{25,100,1000}.dat
# qplot -s 'set logscale xy' -cs 'u 1:4 ps 2' res/res-{LS,SVD}-{25,100,1000}.dat

echo "Existing results are:"
find -path './res/res-*.dat'

echo "Do you want to start experiments?"
if ! ask-yes-no; then
  exit
fi

echo "Do you want to compile?"
if ask-yes-no; then
  x++ -oct -slora leastsq.cpp
fi

NS=(5 10 20 50 100 1000 5000 10000)
Method=(ls svd)
for ((i=0;i<5;i++));do
  for ns in ${NS[@]}; do
    for method in ${Method[@]}; do
      ./a.out $method $ns 5 5 true
      ./a.out $method $ns 10 10 true
      ./a.out $method $ns 25 40 true
    done
  done
done
