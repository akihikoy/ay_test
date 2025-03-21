#!/usr/bin/python3
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time
import sys

i_dxl= int(sys.argv[1]) if len(sys.argv)>1 else 0
dp_trg= int(sys.argv[2]) if len(sys.argv)>2 else 100

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i]) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  dxl[i].CurrentLimit= CURRENT_LIMIT
  dxl[i].Setup()
  dxl[i].EnableTorque()

dxl[i_dxl].MoveTo(dxl[i_dxl].Position()+dp_trg,blocking=True)

print('Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)])


for i,_ in enumerate(DXL_ID):
  #dxl[i].DisableTorque()
  dxl[i].Quit()
