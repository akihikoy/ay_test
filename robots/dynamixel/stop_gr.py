#!/usr/bin/python
#Stop gripper during moving.

from dxl_util import *
from _config import *
import time
import math
import numpy as np

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 2100
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print 'Current position=',dxl.Position()

print 'Type current position, and then hold the gripper to prevent moving'

p_trg= int(raw_input('type target: '))

dxl.MoveTo(p_trg,blocking=False)
for i in range(7):
  time.sleep(0.1)  #wait 0.1 sec
  print 'Current position=',dxl.Position()

print 'Reset the target position to the current position',dxl.Position()
dxl.MoveTo(dxl.Position(),blocking=True)

#dxl.DisableTorque()
dxl.Quit()
