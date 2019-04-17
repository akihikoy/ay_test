#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy
import numpy

env= openravepy.Environment()
env.SetViewer('qtcoin')
env.Load('data2/hiro_test.env.xml')

raw_input("Press Enter to start...")

#robot= env.GetRobots()[0]
robot= env.GetRobot('HiroNX')
robot.SetActiveManipulator('rightarm')
# rightarm, rightarm_torso, leftarm, leftarm_torso, head, head_torso
manip= robot.GetActiveManipulator()
#manip= robot.GetManipulator('leftarm')
# rightarm, rightarm_torso, leftarm, leftarm_torso, head, head_torso
ikmodel = openravepy.databases.inversekinematics.InverseKinematicsModel(robot,iktype=openravepy.IkParameterization.Type.Transform6D)
if not ikmodel.load():
  ikmodel.autogenerate()

#with env: # lock environment
    #Tgoal = numpy.array([[0,-1,0,-0.3],[-1,0,0,0.1],[0,0,-1,0.9],[0,0,0,1]])
    ##Tgoal = numpy.array([[0,-1,0,-0.3],[-1,0,0,-0.05],[0,0,-1,0.9],[0,0,0,1]])
    ##Tgoal = numpy.array([[0,-1,0,-0.23],[-1,0,0,-0.1446],[0,0,-1,0.85],[0,0,0,1]])
    #sol = manip.FindIKSolution(Tgoal, openravepy.IkFilterOptions.CheckEnvCollisions) # get collision-free solution
    #with robot: # save robot state
        #robot.SetDOFValues(sol,manip.GetArmIndices()) # set the current solution
        #Tee = manip.GetEndEffectorTransform()
        #env.UpdatePublishedBodies() # allow viewer to update new robot
        #raw_input('press any key')
    #print Tee

#robot.SetDOFValues([1.5,0,-1.5,0],[9,10,11,12])

manipprob = openravepy.interfaces.BaseManipulation(robot) # create the interface for basic manipulation programs
##Tgoal = numpy.array([[-1,0,0,-0.3],[0,-1,0,-0.08],[0,0,-1,0.9],[0,0,0,1]])
Tgoal = numpy.array([[0,-1,0,-0.27],[-1,0,0,-0.14],[0,0,-1,0.85],[0,0,0,1]])
#Tgoal = numpy.array([[0,-1,0,-0.23],[-1,0,0,-0.1446],[0,0,-1,0.85],[0,0,0,1]])
res = manipprob.MoveToHandPosition(matrices=[Tgoal],seedik=10) # call motion planner with goal joint angles
robot.WaitForController(0) # wait

taskprob = openravepy.interfaces.TaskManipulation(robot) # create the interface for task manipulation programs
taskprob.CloseFingers() # close fingers until collision
robot.WaitForController(0) # wait
with env:
    robot.Grab(env.GetKinBody('mug4'))

goal= numpy.zeros(len(manip.GetArmIndices()))
goal[1]= -1.5
goal[2]= -0.8
manipprob.MoveManipulator(goal)

robot.SetDOFValues([0,0,0,0],[9,10,11,12])

raw_input("Press Enter to exit...")
env.Destroy()
