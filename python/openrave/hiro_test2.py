#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy
import numpy,time

env= openravepy.Environment()
env.SetViewer('qtcoin')
env.Load('data2/hiro_test.env.xml')

logger = env.CreateProblem('logging')
logger.SendCommand('savescene filename testscene.env.xml')

raw_input("Press Enter to start...")

#robot= env.GetRobots()[0]
robot= env.GetRobot('HiroNX')

env.UpdatePublishedBodies()
time.sleep(0.1) # give time for environment to update
#RANDOM POSITIONS:
nav = openravepy.examples.simplenavigation.SimpleNavigationPlanning(robot)
nav.performNavigationPlanning()

raw_input("Press Enter to exit...")
env.Destroy()
