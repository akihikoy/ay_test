#!/usr/bin/python
#\file    find_changept.py
#\brief   Find a change point of a function.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Mar.27, 2021
import scipy.optimize

'''
Find a change point of a function.
Assume a function func(x)={True,False} which has only one change point x0
where func(x)=True if x<x0 else False.
We try to find x0.
'''

if __name__=='__main__':
  def func(x):
    return True if x>0.5 else False

  bounds= [0.0,1.0]

  #Auxiliary function to convert func to a minimization problem.
  def func_opt(x):
    return (x-bounds[0])/(bounds[1]-bounds[0]) if func(x)==True and x>=bounds[0] else 1.0

  #res= scipy.optimize.golden(func_opt, brack=bound, tol=1e-06, maxiter=1000)
  res= scipy.optimize.minimize_scalar(func_opt, bounds=bounds, method='bounded', options={'xatol': 1e-05, 'maxiter': 500})
  print res
  print res.x, func(res.x), func_opt(res.x), func_opt(1), func_opt(0), func_opt(0.55)

