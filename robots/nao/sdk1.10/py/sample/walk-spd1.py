#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *
import time

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)
proxyMo.stiffnessInterpolation('Body', 1.0, 1.0)

x  = 1.0
y  = 0.0
theta  = 0.0
frequency  = 1.0
proxyMo.setWalkTargetVelocity(x, y, theta, frequency)

time.sleep(2)

proxyMo.setWalkTargetVelocity(0.0, 0.0, 0.0, frequency)

