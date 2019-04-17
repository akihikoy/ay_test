#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
import numpy
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('data/lab1.env.xml') # load a simple scene

raw_input("Press Enter to start...")

Ndiv = 10000
Tz = matrixFromAxisAngle([0,0,numpy.pi/4])
dTz = matrixFromAxisAngle([0,0,numpy.pi/4/Ndiv])
with env:
  for i in range(0,Ndiv):
    for body in env.GetBodies():
      body.SetTransform(numpy.dot(dTz,body.GetTransform()))
  env.GetBodies()[0].SetTransform(numpy.dot(Tz,body.GetTransform()))

raw_input("Press Enter to exit...")
env.Destroy()
