#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
import numpy
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
env.Load('data/lab1.env.xml') # load a simple scene

raw_input("Press Enter to start...")

robot=env.GetRobots()[0]
manip = robot.GetActiveManipulator()
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot,iktype=IkParameterization.Type.Transform6D)
if not ikmodel.load():
    ikmodel.autogenerate()

manipprob = interfaces.BaseManipulation(robot) # create the interface for basic manipulation programs
Tgoal = numpy.array([[0,-1,0,-0.23],[-1,0,0,-0.1446],[0,0,-1,0.85],[0,0,0,1]])
res = manipprob.MoveToHandPosition(matrices=[Tgoal],seedik=10) # call motion planner with goal joint angles
robot.WaitForController(0) # wait

taskprob = interfaces.TaskManipulation(robot) # create the interface for task manipulation programs
taskprob.CloseFingers() # close fingers until collision
robot.WaitForController(0) # wait
with env:
    robot.Grab(env.GetKinBody('mug4'))

manipprob.MoveManipulator(numpy.zeros(len(manip.GetArmIndices()))) # move manipulator to all zeros

raw_input("Press Enter to exit...")
env.Destroy()
