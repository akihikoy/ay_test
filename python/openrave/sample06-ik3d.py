#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
env = Environment() # create the environment
env.SetViewer('qtcoin') # start the viewer
env.Load('data/katanatable.env.xml') # load a scene
robot = env.GetRobots()[0] # get the first robot

raw_input("Press Enter to start...")

manip = robot.GetActiveManipulator()
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot,iktype=IkParameterization.Type.Translation3D)
if not ikmodel.load():
    ikmodel.autogenerate()

with robot: # lock environment and save robot state
    robot.SetDOFValues([2.58, 0.547, 1.5, -0.7],[0,1,2,3]) # set the first 4 dof values
    Tee = manip.GetEndEffectorTransform() # get end effector
    ikparam = IkParameterization(Tee[0:3,3],ikmodel.iktype) # build up the translation3d ik query
    sols = manip.FindIKSolutions(ikparam, IkFilterOptions.CheckEnvCollisions) # get all solutions

h = env.plot3(Tee[0:3,3],10) # plot one point
with robot: # save robot state
    for sol in sols[::10]: # go through every 10th solution
        robot.SetDOFValues(sol,manip.GetArmIndices()) # set the current solution
        env.UpdatePublishedBodies() # allow viewer to update new robot
        raw_input('press any key')

print robot.GetDOFValues() # robot state is restored to original

raw_input("Press Enter to exit...")
env.Destroy()
