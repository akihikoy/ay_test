qplot obstacle.dat u 1:2:3 w filledcurves notitle

# GeomPlanningSE2.cpp
qplot -s 'set key center center; set xlabel "x"; set ylabel "y"; set size square' obstacle.dat u 1:2:3 w filledcurves notitle path.dat w lp lt 3 lw 2

# CtrlPlanningSE2.cpp
qplot -s 'set key center center; set xlabel "x"; set ylabel "y"; set size square' obstacle.dat u 1:2:3 w filledcurves notitle path.dat w lp lt 3 lw 2

# GeomPlanningSE2_Info.cpp
qplot -s 'set key center center; set xlabel "x"; set ylabel "y"; set size square' obstacle.dat u 1:2:3 w filledcurves notitle edges.dat w l lt 4 vertices.dat w p lt 2 pt 7 ps 1 path0.dat w lp lt 5 lw 2 path.dat w lp lt 3 lw 2
