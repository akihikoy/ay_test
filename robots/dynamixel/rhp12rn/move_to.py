#!/usr/bin/python3
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 26  #Open
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print('Current position=',dxl.Position())

p_trg= 776  #Close
p_trg= int(input('type target: '))

#Move to a target position
#p_trg= p_start-400
dxl.MoveTo(p_trg)
time.sleep(0.1)  #wait 0.1 sec
print('Current position=',dxl.Position())

#dxl.DisableTorque()
dxl.Quit()
