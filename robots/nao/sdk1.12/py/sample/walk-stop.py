#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)

proxyMo.setWalkTargetVelocity(0.0, 0.0, 0.0, 1.0)

