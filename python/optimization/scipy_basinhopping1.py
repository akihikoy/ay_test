#!/usr/bin/python
#\file    scipy_basinhopping1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.11, 2020

from funcs import Func,Plot
import scipy.optimize
import sys

xmin= [-1.,-1.]
xmax= [2.,3.]
fkind= int(sys.argv[1]) if len(sys.argv)>1 else 2
f= lambda x:Func(x,fkind)

# Minimize the function f
accept_test= lambda **kwargs: all(x0<=x and x<=x1 for x,x0,x1 in zip(kwargs['x_new'],xmin,xmax))
res= scipy.optimize.basinhopping(f, [0.0,0.0], niter=100, T=1.0, stepsize=0.5, accept_test=accept_test)

print res
print 'Result=',res.x,accept_test(x_new=res.x)

Plot(xmin,xmax,f,x_points=[res.x])
