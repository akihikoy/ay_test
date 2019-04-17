#!/usr/bin/python

# Importing libraries...
from scipy.optimize import minimize

# Objective function to be minimized
def f(x):
  return (x[0]-2.0)**2 + (x[1]-3.0)**2

# Minimize the function f
res= minimize(f,[0.0,0.0],bounds=[[-1.0,1.0],[-1.0,1.0]])

print res
print 'Result=',res.x
