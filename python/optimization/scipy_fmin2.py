#!/usr/bin/python3

from funcs import Func,Plot
import scipy.optimize
import sys

xmin= [-1.,-1.]
xmax= [2.,3.]
fkind= int(sys.argv[1]) if len(sys.argv)>1 else 0
f= lambda x:Func(x,fkind)

# Minimize the function f
res= scipy.optimize.minimize(f,[0.0,0.0],bounds=[[-1.0,1.0],[-1.0,1.0]])

print(res)
print('Result=',res.x)

Plot(xmin,xmax,f,x_points=[res.x])
