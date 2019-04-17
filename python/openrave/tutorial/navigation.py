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

# test to change the robot initial position:
R = robot.GetTransform()
print R
R[0,3]= 2
R[1,3]= 2
R[2,3]= 0
robot.SetTransform(R)
raw_input("Press Enter again")


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

from mynavigation import MyNavigationPlanning
env.UpdatePublishedBodies()
time.sleep(0.1)
nav = MyNavigationPlanning(robot)
nav.moveTo([1.5,1.5,numpy.pi*0.25])
nav.moveTo([1.0,-1.5,-numpy.pi*0.5])

raw_input("Press Enter to exit...")
env.Destroy()
