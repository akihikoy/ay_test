#!/usr/bin/python
#Change ID.

from dxl_util import *

DXL_ID= 1  #Current ID
NEW_ID= 2  #New ID
BAUDRATE= 57600
#DXL_TYPE= 'XM430-W350'  #Finger robot
DXL_TYPE= 'XH430-V350'  #Dynamixel gripper

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()

#Change ID
dxl.Write('ID',NEW_ID)

dxl.Quit()
