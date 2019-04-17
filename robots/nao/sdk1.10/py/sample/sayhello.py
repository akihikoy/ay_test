#!/usr/bin/python
# -*- coding: utf-8 -*-
from naoqi import ALProxy
from naoconfig import *

proxyAudio = ALProxy("ALTextToSpeech",robot_IP,robot_port)
proxyAudio.say("Hello world")
