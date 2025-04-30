#!/usr/bin/python3
#Move Dynamixel to the initial position, and to a target

from dxl_util import *
from _config import *
import time
import sys
from kbhit2 import KBHAskGen

#i_dxl= int(sys.argv[1]) if len(sys.argv)>1 else 0
#dp_trg= int(sys.argv[2]) if len(sys.argv)>2 else 100

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i]) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  #dxl[i].CurrentLimit= CURRENT_LIMIT
  dxl[i].Setup()
  dxl[i].EnableTorque()

#dxl[i_dxl].MoveTo(dxl[i_dxl].Position()+dp_trg,blocking=True)

pos_trg=[
  [1550, 1624, 1800],
  [1150, 3300],
  ]
i_pos_trg= [0, 0]

while True:
  i_dxl= -1
  key= KBHAskGen('a','s','q')
  if key=='q':
    break
  elif key=='a':
    i_dxl= 0
  elif key=='s':
    i_dxl= 1

  if i_dxl>=0:
    p_trg= pos_trg[i_dxl][i_pos_trg[i_dxl]]
    dxl[i_dxl].MoveTo(p_trg,blocking=False)
    i_pos_trg[i_dxl]+= 1
    if i_pos_trg[i_dxl]>=len(pos_trg[i_dxl]):
      i_pos_trg[i_dxl]= 0

  print('Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)])


for i,_ in enumerate(DXL_ID):
  dxl[i].DisableTorque()
  dxl[i].Quit()
