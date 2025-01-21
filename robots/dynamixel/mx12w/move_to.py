#!/usr/bin/python3
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time
import sys

dp_trg= int(sys.argv[1]) if len(sys.argv)>1 else 100

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i],dev=DXL_DEV) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= 'JOINT'
  dxl[i].Setup()
  dxl[i].EnableTorque()

dxl[0].MoveTo(dxl[0].Position()+dp_trg,blocking=True)

print('Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)])


for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
