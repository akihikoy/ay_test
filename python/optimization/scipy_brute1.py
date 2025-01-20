#!/usr/bin/python3
#\file    scipy_brute1.py
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
res= scipy.optimize.brute(f, np.array([xmin,xmax]).T, Ns=5)

print('Result=',res)

Plot(xmin,xmax,f,x_points=[res])
