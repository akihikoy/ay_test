#!/usr/bin/python
import math

#Modify angle to [-pi,pi]
def ModAngle(angle):
  while angle<-math.pi:  angle+= 2.0*math.pi
  while angle>math.pi:   angle-= 2.0*math.pi
  return angle

#Displacement of two angles, whose absolute value is less than pi
def AngleDisplacement(angle1, angle2):
  angle1= ModAngle(angle1)
  angle2= ModAngle(angle2)
  if angle2>=angle1:
    d= angle2-angle1
    return d if d<=math.pi else d-2.0*math.pi
  else:
    d= angle1-angle2
    return -d if d<=math.pi else 2.0*math.pi-d

#Float version of range
def FRange1(xmin,xmax,num_div):
  return [xmin+(xmax-xmin)*x/float(num_div) for x in range(num_div+1)]


angles= FRange1(-math.pi*1.2, math.pi*1.2, 20)
for i in range(len(angles)-1):
  print AngleDisplacement(angles[i],angles[i+1]), AngleDisplacement(angles[i+1],angles[i])

