#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
import numpy
env = Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.Load('data/pr2test1.env.xml') # load a scene
robot = env.GetRobots()[0] # get the first robot

raw_input("Press Enter to start...")

robot.SetActiveManipulator('leftarm_torso') # set the manipulator to leftarm + torso
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot,iktype=IkParameterization.Type.Transform6D)
if not ikmodel.load():
    ikmodel.autogenerate()

manipprob = interfaces.BaseManipulation(robot) # create the interface for basic manipulation programs
Tgoal = numpy.array([[0,-1,0,-0.21],[-1,0,0,0.04],[0,0,-1,0.92],[0,0,0,1]])
res = manipprob.MoveToHandPosition(matrices=[Tgoal],seedik=10) # call motion planner with goal joint angles
robot.WaitForController(0) # wait

raw_input("Press Enter to exit...")
env.Destroy()
