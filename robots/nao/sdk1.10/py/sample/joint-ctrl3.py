#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)
proxyMo.stiffnessInterpolation('Body', 1.0, 1.0)

names = ['HeadYaw','HeadPitch']
angleLists = [[1.2], [-0.3]]
timeLists = [[1.0], [1.0]]
proxyMo.angleInterpolation(names, angleLists, timeLists, True)

