#!/usr/bin/python
#Control Dynamixel to follow a sin curve (current-based PD control)

from dxl_util import *
from _config import *
import time
import math
import numpy as np

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.OpMode= 'CURRENT'
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 2000
#dxl.MoveTo(p_start)
#time.sleep(2.0)  #wait 2 sec

tol= 5
#kp,kd= 50.0, 5.0
#kp,kd= 50.0, 1.0
kp,kd= 0.5, 0.2
while True:
  curr= kp*(p_start-dxl.Position()) + kd*(0.0-dxl.Velocity())
  print curr
  dxl.SetCurrent(int(curr))
  if abs(p_start-dxl.Position())<tol:
    dxl.SetCurrent(0)
    break

print 'Current position=',dxl.Position()

#Move to a target position
for t in np.mgrid[0:2*math.pi:0.01]:
  p_trg= p_start + 250*(0.5-0.5*math.cos(t))
  #print p_trg
  curr= kp*(p_trg-dxl.Position()) + kd*(0.0-dxl.Velocity())
  print curr
  dxl.SetCurrent(int(curr))
  time.sleep(0.001)
  #print 'Current position=',dxl.Position()

dxl.DisableTorque()
dxl.Quit()
