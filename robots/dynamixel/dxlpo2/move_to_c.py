#!/usr/bin/python
#Move Dynamixel to the initial position, and to a target position and current

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.CurrentLimit= CURRENT_LIMIT
#dxl.OpMode= 'CURRPOS'  #WARNING: This control mode is not available with this Dynamixel.
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 0  #Open
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print 'Current position=',dxl.Position()
print 'Current current=',dxl.Current()

#p_trg= -250962  #Close
#p_trg= -200000
p_trg= -150000
#p_trg= -125000  #Half-close
#p_trg= int(raw_input('type target: '))
#c_trg= 2000  #Max current
c_trg= 1000
#c_trg= int(raw_input('type current: '))

#Move to a target position
dxl.MoveToC(p_trg, c_trg)
time.sleep(0.1)  #wait 0.1 sec
print 'Current position=',dxl.Position()
print 'Current current=',dxl.Current()

#dxl.DisableTorque()
dxl.Quit()
