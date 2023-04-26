#!/usr/bin/python
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.CurrentLimit= CURRENT_LIMIT
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= dxl.Position()
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print 'Current position=',dxl.Position()

raw_input('hit enter to continue: ')

#p_trg= -250962  #Close
#p_trg= -200000
#p_trg= -125000  #Half-close
#p_trg= int(raw_input('type target: '))
p_trg= -19000

#Move to a target position
dxl.MoveTo(p_trg)
time.sleep(0.1)  #wait 0.1 sec
print 'Current position=',dxl.Position()

raw_input('hit enter to go back: ')

dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print 'Current position=',dxl.Position()


#dxl.DisableTorque()
dxl.Quit()
