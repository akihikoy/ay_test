#!/usr/bin/python3
#Control Dynamixel to follow a sin curve (with current-based position control mode)

from dxl_util import *
from _config import *
import time
import math
import numpy as np

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.OpMode= 'CURRPOS'
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 2000
dxl.MoveTo(p_start)
time.sleep(2.0)  #wait 2 sec
print('Current position=',dxl.Position())

#Move to a target position
for t in np.mgrid[0:6*math.pi:0.05]:
  p_trg= p_start + 250*(0.5-0.5*math.cos(t))
  #print p_trg
  curr= dxl.CurrentLimit if (dxl.Position()-p_trg)>0 else -dxl.CurrentLimit
  #curr= 10
  curr= 100
  dxl.MoveToC(p_trg, current=curr, blocking=False)
  time.sleep(0.01)
  #print 'Current position=',dxl.Position()

dxl.DisableTorque()
dxl.Quit()
