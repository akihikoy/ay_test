#!/usr/bin/python
import math

# Matlab-like mod function that returns always positive
def Mod(x, y):
  if y==0:  return x
  return x-y*math.floor(x/y)


x= -10
while x<10:
  y= Mod(x,math.pi)
  print x,y
  x+=0.01

