#!/bin/bash

cd dat-cls1
tmpf=/tmp/plot$$
paste features.dat labels.dat | awk '{if($3==1) print $0}' > $tmpf-1.dat
paste features.dat labels.dat | awk '{if($3==-1) print $0}' > $tmpf-2.dat
paste features.dat out_labels.dat | awk '{if($3>0.1) print $0}' > $tmpf-3.dat
paste features.dat out_labels.dat | awk '{if($3<-0.1) print $0}' > $tmpf-4.dat

# qplot \
qplot -3d \
  $tmpf-1.dat lt 1 pt 4 \
  $tmpf-2.dat lt 3 pt 4 \
  $tmpf-3.dat lt 1 pt 1 \
  $tmpf-4.dat lt 3 pt 1

echo ''
echo -en 'deleating tmp files... '
read s
rm ${tmpf}*
