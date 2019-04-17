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
frequency  = 0.6
proxyMo.setWalkTargetVelocity(x, y, theta, frequency)
time.sleep(2)
# 前方に歩行(2秒間)

x = 0.0
theta = 0.5
proxyMo.setWalkTargetVelocity(x, y, theta, frequency)
time.sleep(2)
# 左方向に回転(2秒間)

x = 1.0
theta = -0.5
proxyMo.setWalkTargetVelocity(x, y, theta, frequency)
time.sleep(5)
# 前進しながら右方向に回転(5秒間)

x  = 1.0
theta  = 0.0
frequency  = 1.0
proxyMo.setWalkTargetVelocity(x, y, theta, frequency)
time.sleep(5)
# やや早く前進(5秒間)

proxyMo.setWalkTargetVelocity(0.0, 0.0, 0.0, frequency)
# 速度ゼロ(ストップ)
