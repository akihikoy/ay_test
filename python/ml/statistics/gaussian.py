#!/usr/bin/python3
from math import *
import numpy

mean= 7.0
stddev= 0.8
var= stddev*stddev

xmin= -5
xmax= 20
ndiv= 1000.0
dx= (xmax-xmin)/ndiv


def Square(x):
  return x*x

def Probability(x):
  return exp(-Square(x-mean)/(2.0*var)) / sqrt(pi*var)

for x in numpy.arange(xmin,xmax,dx):
  print(str(x)+" "+str(Probability(x)))

