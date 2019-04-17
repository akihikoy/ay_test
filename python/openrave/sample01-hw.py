#!/usr/bin/python
# -*- coding: utf-8 -*-
from openravepy import *
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('data/lab1.env.xml') # load a simple scene
robot = env.GetRobots()[0] # get the first robot

with env: # lock the environment since robot will be used
    print "Robot ",robot.GetName()," has ",robot.GetDOF()," joints with values:\\n",robot.GetJointValues()
    robot.SetDOFValues([0.5],[0]) # set joint 0 to value 0.5
    T = robot.GetLinks()[1].GetTransform() # get the transform of link 1
    print "The transformation of link 1 is:\n",T

raw_input("Press Enter to exit...")
env.Destroy()

