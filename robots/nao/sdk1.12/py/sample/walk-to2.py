#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)
proxyMo.stiffnessInterpolation('Body', 1.0, 1.0)

x = 0.5
y = 0.0
theta = -0.3
proxyMo.walkTo(x, y, theta)

#x = -0.2
#y = 0.0
#theta  = 0.0
#proxyMo.walkTo(x, y, theta)
## 後ろに 0.2 [m] 下がる

#x = 0.0
#y = 0.0
#theta  = -1.5709
#proxyMo.walkTo(x, y, theta)
## 右に90度回転
