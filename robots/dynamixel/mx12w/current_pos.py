#!/usr/bin/python
#Printing current position without enabling torque.

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i],dev=DXL_DEV) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  dxl[i].Setup()
  #dxl[i].EnableTorque()

try:
  f= 0
  t0= time.time()
  while True:
    print 'Positions=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)]
    f+= 1
    #time.sleep(0.001)
except KeyboardInterrupt:
  print 'Observation freq:',f/(time.time()-t0),'Hz'

for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
