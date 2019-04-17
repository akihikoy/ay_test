#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)
proxyMo.stiffnessInterpolation('Body', 1.0, 1.0)

proxyMo.angleInterpolation(['LHand','RHand'], [[1.0],[1.0]], [[1.0],[1.0]], True)
proxyMo.angleInterpolation(['LHand','RHand'], [[0.0],[0.0]], [[1.0],[1.0]], True)
proxyMo.angleInterpolation(['LHand','RHand'], [[0.5],[0.5]], [[1.0],[1.0]], True)
