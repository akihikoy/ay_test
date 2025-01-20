#!/usr/bin/python3
#\file    scipy_solve_1d_eq.py
#\brief   Comparing method for solving 1d variable equation.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.21, 2021
import scipy.optimize
import time

f= lambda x: x+x**2+x**3 - 2.5
bounds= [-1.0,1.0]
#bounds= [-1.0,0.5]  #NOTE: Setup where the solution is not within the bounds.

#We will solve x for f(x)==0 s.t. x in bounds.

results= []

#Using scipy.optimize.minimize.
t_start= time.time()
res= scipy.optimize.minimize(lambda x:f(x[0])**2,[0.0],bounds=[bounds])
results.append(['optimize.minimize\t\t',res.x[0],f(res.x[0]),time.time()-t_start])

#Using scipy.optimize.minimize_scalar.  #WARNING: Does not consider bounds.
t_start= time.time()
res= scipy.optimize.minimize_scalar(lambda x:f(x)**2, bracket=bounds, method='brent')
results.append(['optimize.minimize_scalar-brent',res.x,f(res.x),time.time()-t_start])

#Using scipy.optimize.minimize_scalar.  #WARNING: Does not consider bounds.
t_start= time.time()
res= scipy.optimize.minimize_scalar(lambda x:f(x)**2, bracket=bounds, method='golden')
results.append(['optimize.minimize_scalar-golden',res.x,f(res.x),time.time()-t_start])

#Using scipy.optimize.minimize_scalar.
t_start= time.time()
res= scipy.optimize.minimize_scalar(lambda x:f(x)**2, bounds=bounds, method='bounded')
results.append(['optimize.minimize_scalar-bounded',res.x,f(res.x),time.time()-t_start])

#Using scipy.optimize.fsolve.
t_start= time.time()
res= scipy.optimize.fsolve(lambda x:f(x[0]),[0.0])  #WARNING: Does not consider bounds.
results.append(['optimize.fsolve\t\t\t',res[0],f(res[0]),time.time()-t_start])


print('method\t\t\t\t\t x\t\t f(x)\t\t\t time')
for method,x,f_x,t in results:
  print(method,':\t',x,'\t',f_x,'\t',t)

