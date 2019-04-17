#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyMo = ALProxy('ALMotion',robot_IP,robot_port)

proxyMo.post.openHand('LHand')
proxyMo.openHand('RHand')

proxyMo.post.closeHand('LHand')
proxyMo.closeHand('RHand')
