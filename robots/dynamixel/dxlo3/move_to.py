#!/usr/bin/python
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= [TDynamixel1(DXL_TYPE),TDynamixel1(DXL_TYPE)]
for i in range(2):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()
  dxl[i].EnableTorque()

#Function to move dp (offset from close)
def MoveDp(dp):
  p_trg= [2048+dp,2048-dp]
  dxl[0].MoveTo(p_trg[0],blocking=False)
  dxl[1].MoveTo(p_trg[1],blocking=True)
  print 'Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)]

#Move to initial position
#MoveDp(260)  #Open
MoveDp(400)  #Open
time.sleep(0.5)  #wait .5 sec

raw_input('Move? > ')
#MoveDp(0)  #Close(without fingertips)
MoveDp(108)  #Close(with FV+)


for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
