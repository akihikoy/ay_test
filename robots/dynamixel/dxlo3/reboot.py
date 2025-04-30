#!/usr/bin/python3
#Reboot Dynamixel.

from dxl_util import *
from _config import *

#Setup the device
dxl= [TDynamixel1(DXL_TYPE),TDynamixel1(DXL_TYPE)]
for i in range(2):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()
  dxl[i].EnableTorque()

print('Rebooting Dynamixel...')
for i in range(2):
  dxl[i].Reboot()

for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
