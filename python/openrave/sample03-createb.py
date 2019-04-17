#!/usr/bin/python
# -*- coding: utf-8 -*-

from openravepy import *
import numpy, time
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)

raw_input("Press Enter to start...")

with env:
    body = RaveCreateKinBody(env,'')
    body.SetName('testbody')
    body.InitFromBoxes(numpy.array([[0,0,0,0.1,0.2,0.3]]),True) # set geometry as one box of extents 0.1, 0.2, 0.3
    env.AddKinBody(body)

time.sleep(4) # sleep 4 seconds
with env:
    env.Remove(body)
    body.InitFromBoxes(numpy.array([[-0.4,0,0,0.1,0.2,0.3],[0.4,0,0,0.1,0.2,0.9]]),True) # set geometry as two boxes
    env.AddKinBody(body)

raw_input("Press Enter to exit...")
env.Destroy()
