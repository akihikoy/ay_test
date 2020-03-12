#!/usr/bin/python
#\file    scipy_differential_evolution1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020

from funcs import Func,Plot
import scipy.optimize
import sys
import numpy as np

xmin= [-1.,-1.]
xmax= [2.,3.]
fkind= int(sys.argv[1]) if len(sys.argv)>1 else 2
f= lambda x:Func(x,fkind)

# Minimize the function f
tol= 1.0e-5
res= scipy.optimize.differential_evolution(f, np.array([xmin,xmax]).T, strategy='best1bin', maxiter=300, popsize=10, tol=tol, mutation=(0.5, 1), recombination=0.7)

print res
print 'Result=',res.x

Plot(xmin,xmax,f,x_points=[res.x])
