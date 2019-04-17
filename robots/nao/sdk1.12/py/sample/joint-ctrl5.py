#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)
proxyMo.stiffnessInterpolation('Body', 1.0, 1.0)

angleLists = [[0.0,1.0], [0.0,1.0], [0.0,-1.0], [0.0,-1.0], [0.0,-1.0], [0.0,1.0]]
timeLists = [[1.0,2.0]]*6
proxyMo.post.angleInterpolation('LArm', angleLists, timeLists, True)

angleLists = [[0.0,-1.0], [0.0,-1.0], [0.0,1.0], [0.0,1.0], [0.0,1.0], [0.0,1.0]]
timeLists = [[1.0,2.0]]*6
proxyMo.angleInterpolation('RArm', angleLists, timeLists, True)
