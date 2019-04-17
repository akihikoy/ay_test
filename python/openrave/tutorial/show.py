#!/usr/bin/python
# -*- coding: utf-8 -*-

import openravepy

env = openravepy.Environment()
env.SetViewer('qtcoin')
env.Load('env1.xml')
#robot= env.GetRobots()[0]
robot= env.GetRobot('SimpleBot')

raw_input("Press Enter to exit...")
env.Destroy()
