#!/usr/bin/python3
#Move Dynamixel to the initial position, and to a target position and current

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
dxl.OpMode= 'CURRPOS'
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 2100
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print('Current position=',dxl.Position())
print('Current current=',dxl.Current())

p_trg= int(input('type target: '))
c_trg= int(input('type current: '))

#Move to a target position
#p_trg= p_start-400
dxl.MoveToC(p_trg, c_trg)
time.sleep(0.1)  #wait 0.1 sec
print('Current position=',dxl.Position())
print('Current current=',dxl.Current())

#dxl.DisableTorque()
dxl.Quit()
