#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy
import numpy,time

env = openravepy.Environment()
env.SetViewer('qtcoin')
env.Load('env1.xml')
#robot= env.GetRobots()[0]
robot= env.GetRobot('SimpleBot')

raw_input("Press Enter to start...")

for i in range(0,10):
  body = openravepy.RaveCreateKinBody(env,'')
  body.SetName('obstacle'+str(i))
  boxinfo= numpy.array([[0,0,0.0, 0.1,0.15,0.3]])
  render= True
  body.InitFromBoxes(boxinfo,render)
  R = openravepy.matrixFromAxisAngle([0,0,2.0*numpy.pi*numpy.random.rand()])
  R[0,3]= 4.5*(numpy.random.rand()-0.5)
  R[1,3]= 4.5*(numpy.random.rand()-0.5)
  R[2,3]= 0.3
  print R
  body.SetTransform(R)
  env.AddKinBody(body)

env.UpdatePublishedBodies()
time.sleep(0.1)
nav = openravepy.examples.simplenavigation.SimpleNavigationPlanning(robot)
nav.performNavigationPlanning()

raw_input("Press Enter to exit...")
env.Destroy()
