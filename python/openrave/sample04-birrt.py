#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
env = Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.Load('data/lab1.env.xml') # load a scene
robot = env.GetRobots()[0] # get the first robot

raw_input("Press Enter to start...")

manipprob = interfaces.BaseManipulation(robot) # create the interface for basic manipulation programs
res = manipprob.MoveManipulator(goal=[-0.75,1.24,-0.064,2.33,-1.16,-1.548,1.19]) # call motion planner with goal joint angles
robot.WaitForController(0) # wait

raw_input("Press Enter to exit...")
env.Destroy()
