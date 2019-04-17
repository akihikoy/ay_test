#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
import numpy, time
env = Environment()
env.SetViewer('qtcoin') # start the viewer
env.Load('data/pr2test1.env.xml')

raw_input("Press Enter to start...")

robot=env.GetRobots()[0]
manip = robot.SetActiveManipulator('leftarm_torso')
lower,upper = robot.GetDOFLimits(manip.GetArmIndices()) # get the limits of just the arm indices
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot,iktype=IkParameterization.Type.Transform6D)
if not ikmodel.load():
    ikmodel.autogenerate()

maxtime = 0.1 # 100ms
while True:
    with env:
        robot.SetDOFValues(lower+numpy.random.rand(len(lower))*(upper-lower),manip.GetArmIndices()) # set a random values to just the arm
        incollision = not env.CheckCollision(robot) and not robot.CheckSelfCollision()
        starttime = time.time()
        def timeoutfilter(values, manip, ikparam):
            return IkFilterReturn.Quit if time.time()-starttime > maxtime else IkFilterReturn.Success

        manip.GetIkSolver().SetCustomFilter(timeoutfilter)
        success = manip.FindIKSolution(manip.GetIkParameterization(IkParameterization.Type.Transform6D),IkFilterOptions.CheckEnvCollisions)
        print 'in collision: %d, real success: %d, time passed: %f'%(incollision,success is not None,time.time()-starttime)

raw_input("Press Enter to exit...")
env.Destroy()
