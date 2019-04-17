#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy

def test(robot_IP,is_simulation):

  #if not is_simulation:
    #audioProxy = ALProxy("ALTextToSpeech",robot_IP,9559)
    #audioProxy.post.say("I am Tanaka")  # .post make a parallel call.

  #memProxy = ALProxy("ALMemory",robot_IP,9559)
  #memProxy.insertData("myValueName", 0)

  proxyMo = ALProxy('ALMotion',robot_IP,9559)

  # Example showing how to interpolate to maximum stiffness in 1 second
  names  = 'Body'
  stiffnessLists  = 1.0  # NOTE: it seems not working in Choregraphe
  timeLists  = 1.0
  proxyMo.stiffnessInterpolation(names, stiffnessLists, timeLists)

  # Example showing a single target angle for one joint
  # Interpolate the head yaw to 1.0 radian in 1.0 second
  names  = ['HeadYaw', 'HeadPitch']
  # angles  = [[1.0], [0.2]]
  # times = [[1.0], [1.0]]
  angles  = [[1.0, 0.0], [-0.5, 0.5, 0.0]]
  times   = [[1.0, 2.0], [ 1.0, 2.0, 3.0]]
  isAbsolute  = True
  proxyMo.angleInterpolation(names, angles, times, isAbsolute)


  # Example showing how to set angles, using a fraction of max speed
  #names  = ['HeadYaw', 'HeadPitch']
  #angles  = [-1.0, -0.2]
  #fractionMaxSpeed  = 0.2
  #proxyMo.setAngles(names, angles, fractionMaxSpeed)

# NOTE: does not work in Choregraphe
def switch_servo(robot_IP,stiff):

  proxyMo = ALProxy('ALMotion',robot_IP,9559)

  # Example showing how to interpolate to maximum stiffness in 1 second
  names  = 'Body'
  stiffnessLists  = stiff
  timeLists  = 1.0
  proxyMo.stiffnessInterpolation(names, stiffnessLists, timeLists)

def servo_on(robot_IP):
  switch_servo(robot_IP,1.0)

def servo_off(robot_IP):
  switch_servo(robot_IP,0.0)

