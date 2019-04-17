#!/bin/bash

setting=""
setting+="set xlabel 'Number of samples';"
setting+="set ylabel 'Computation time';"
setting+="set xrange [4:11000];"
setting+="set yrange [*:500];"
setting+="set logscale xy;"
setting+="set key left top;"

exp=$(dirname $0)
figname=$exp/fig-time.svg
u="u 1:5"
plots=""
plots+=" $exp/res-LS-25.dat    $u lt 1 pt 1 ps 3"
plots+=" $exp/res-LS-100.dat   $u lt 1 pt 2 ps 3"
plots+=" $exp/res-LS-1000.dat  $u lt 1 pt 3 ps 3"
plots+=" $exp/res-SVD-25.dat   $u lt 3 pt 4 ps 2"
plots+=" $exp/res-SVD-100.dat  $u lt 3 pt 6 ps 2"
plots+=" $exp/res-SVD-1000.dat $u lt 3 pt 8 ps 2"

qplot -s "$setting" $plots
echo "save graph into $figname ?"
if ask-yes-no; then
  qplot -s "$setting" $plots -o $figname
fi

