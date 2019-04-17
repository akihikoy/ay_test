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
  boxinfo= numpy.array([[0,0,0.3, 0.1,0.15,0.3]])
  boxinfo[0,0]= 4.5*(numpy.random.rand()-0.5)
  boxinfo[0,1]= 4.5*(numpy.random.rand()-0.5)
  render= True
  print boxinfo
  body.InitFromBoxes(boxinfo,render)
  env.AddKinBody(body)

raw_input("Press Enter to exit...")
env.Destroy()
