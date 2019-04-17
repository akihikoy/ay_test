#!/bin/bash

cd dat-1cls1
tmpf=/tmp/plot$$
paste test_features.dat out_labels.dat | awk '{if($1=="" || $3>0.1) print $0}' > $tmpf-1.dat
paste test_features.dat out_labels.dat | awk '{if($1=="" || $3<-0.1) print $0}' > $tmpf-2.dat

# qplot \
qplot -3d \
  features.dat u '1:2:(0)' lt 1 pt 5 \
  $tmpf-1.dat lt 1 pt 1 \
  $tmpf-2.dat lt 3 pt 1

echo ''
echo -en 'deleating tmp files... '
read s
rm ${tmpf}*
