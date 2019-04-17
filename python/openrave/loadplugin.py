#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy
import os

env = openravepy.Environment()
env.SetViewer('qtcoin')
env.Load('tutorial/env1.xml')
#robot= env.GetRobots()[0]
robot= env.GetRobot('SimpleBot')

#openravepy.RaveLoadPlugin(os.getcwd()+'/libtestcontrollers.so')
openravepy.RaveLoadPlugin(os.getcwd()+'/test_ctrl/build/libtestcontrollers.so')
controller= openravepy.RaveCreateController(env,'TestController')
robot.SetController(controller,range(robot.GetDOF()),controltransform=1)

raw_input("Press Enter to start...")

import time
env.UpdatePublishedBodies()
time.sleep(0.1)
nav = openravepy.examples.simplenavigation.SimpleNavigationPlanning(robot)
nav.performNavigationPlanning()

raw_input("Press Enter to exit...")
env.Destroy()
