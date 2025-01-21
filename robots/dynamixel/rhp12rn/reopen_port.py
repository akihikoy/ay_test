#!/usr/bin/python3
#Port reopen test.

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
p_trg= dxl.Position()+60
dxl.MoveTo(p_trg)
print('Current position=',dxl.Position())

print('Closing port...')
dxl.Quit()

input('Type anything to continue > ')
#time.sleep(0.1)
print('Reopening port...')
dxl.Setup()
dxl.EnableTorque()

p_trg= p_trg-20
dxl.MoveTo(p_trg)
print('Current position=',dxl.Position())
time.sleep(0.1)

input('Type anything to continue > ')

p_trg= p_trg-20
dxl.MoveTo(p_trg)
print('Current position=',dxl.Position())

input('Type anything to continue > ')

p_trg= p_trg-20
dxl.MoveTo(p_trg)
print('Current position=',dxl.Position())

input('Type anything to continue > ')

p_trg= p_trg-20
dxl.MoveTo(p_trg)
print('Current position=',dxl.Position())

dxl.DisableTorque()
dxl.Quit()
