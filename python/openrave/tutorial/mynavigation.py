#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy
import time,numpy

class MyNavigationPlanning:
  def __init__(self,robot):
    self.env = robot.GetEnv()
    self.robot = robot
    self.cdmodel = openravepy.databases.convexdecomposition.ConvexDecompositionModel(self.robot)
    if not self.cdmodel.load():
      self.cdmodel.autogenerate()
    self.basemanip = openravepy.interfaces.BaseManipulation(self.robot)

  def moveTo(self,goal,envmin=[-2.5,-2.5,0],envmax=[2.5,2.5,1]):
    with self.env:
      self.robot.SetAffineTranslationLimits(envmin,envmax)
      self.robot.SetAffineTranslationMaxVels([0.5,0.5,0.5])
      self.robot.SetAffineRotationAxisMaxVels(numpy.ones(4))
      self.robot.SetActiveDOFs([],openravepy.Robot.DOFAffine.X|openravepy.Robot.DOFAffine.Y|openravepy.Robot.DOFAffine.RotationAxis,[0,0,1])
    with self.env:
      with self.robot:
        self.robot.SetActiveDOFValues(goal)
        if self.env.CheckCollision(self.robot):
          print 'invalid goal is specified: ',goal
          return False
    print 'planning to: ',goal
    # draw the marker
    center = numpy.r_[goal[0:2],0.2]
    xaxis = 0.5*numpy.array((numpy.cos(goal[2]),numpy.sin(goal[2]),0))
    yaxis = 0.25*numpy.array((-numpy.sin(goal[2]),numpy.cos(goal[2]),0))
    h = self.env.drawlinelist(numpy.transpose(numpy.c_[center-xaxis,center+xaxis,center-yaxis,center+yaxis,center+xaxis,center+0.5*xaxis+0.5*yaxis,center+xaxis,center+0.5*xaxis-0.5*yaxis]),linewidth=5.0,colors=numpy.array((0,1,0)))
    # start to move
    if self.basemanip.MoveActiveJoints(goal=goal,maxiter=3000,steplength=0.1,outputtraj=True) is None:
      print 'failed to plan to the goal: ',goal
      return False
    print 'waiting for controller'
    self.robot.WaitForController(0)
    return True

