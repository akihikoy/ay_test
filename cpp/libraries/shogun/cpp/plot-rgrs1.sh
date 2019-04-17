#!/bin/bash

cd dat-rgrs1
tmpf=/tmp/plot$$
paste features.dat labels.dat > $tmpf-1.dat
paste test_features.dat out_labels.dat > $tmpf-2.dat

# qplot \
qplot -3d \
  $tmpf-1.dat lt 1 pt 5 \
  $tmpf-2.dat w l lt 3

echo ''
echo -en 'deleating tmp files... '
read s
rm ${tmpf}*
