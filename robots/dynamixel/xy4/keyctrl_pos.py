#!/usr/bin/python3
#Control dynamixel with key input (position control).
#NOTE: Run before this script: rosrun ay_util fix_usb_latency.sh
#NOTE: Run before this script: ../fix_usb_latency.sh

'''
Keyboard interface:
  'q':  Quit.
  'd','a','w','s': Right turn, left turn, up, down.
  'C','B': Move to Center (of rotation), Base (height).
'''

from dxl_util import *
from _config import *
import time

from kbhit2 import TKBHit
import threading
import sys

#Setup the device
dxl= [TDynamixel1(DXL_TYPE[i]) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  dxl[i].CurrentLimit= CURRENT_LIMIT
  dxl[i].Setup()
  dxl[i].EnableTorque()

#Move to the current position
p_start= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].MoveTo(p_start[i])
time.sleep(0.5)  #wait .5 sec
print('Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)])


def ReadKeyboard(is_running, key_cmd, key_locker):
  dt_hold= 0.1
  t_prev= 0
  with TKBHit() as kbhit:
    while is_running[0]:
      c= kbhit.KBHit()
      if c is not None or time.time()-t_prev>dt_hold:
        with key_locker:
          key_cmd[0]= c
        t_prev= time.time()
      time.sleep(0.0025)

key_cmd= [None]
key_locker= threading.RLock()
is_running= [True]
t1= threading.Thread(name='t1', target=lambda a1=is_running,a2=key_cmd,a3=key_locker: ReadKeyboard(a1,a2,a3))
t1.start()

trg= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]  #Current position
init= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= [0.0, 0.0, 0.0, 0.0]
  d= [200*e for e in [1,1,1,1]]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','a','s', 'Z','X','A','S', 'c','C', 'v','V', 'd','D', 'f','F'):
      mov= {
        'z':[d[0],0.0,0.0,0.0],
        'x':[0.0,d[0],0.0,0.0],
        'a':[0.0,0.0,d[0],0.0],
        's':[0.0,0.0,0.0,d[0]],
        'Z':[-d[0],0.0,0.0,0.0],
        'X':[0.0,-d[0],0.0,0.0],
        'A':[0.0,0.0,-d[0],0.0],
        'S':[0.0,0.0,0.0,-d[0]],
        'c':[d[0],d[0],0.0,0.0],
        'C':[-d[0],-d[0],0.0,0.0],
        'v':[d[0],-d[0],0.0,0.0],
        'V':[-d[0],d[0],0.0,0.0],
        'd':[0.0,0.0,d[0],d[0]],
        'D':[0.0,0.0,-d[0],-d[0]],
        'f':[0.0,0.0,d[0],-d[0]],
        'F':[0.0,0.0,-d[0],d[0]],
        }[c]
    elif c=='I':
      trg= init
      for i,_ in enumerate(DXL_ID):
        dxl[i].MoveTo(int(trg[i]), blocking=False)
    elif c=='r':
      for i,_ in enumerate(DXL_ID):
        dxl[i].Reboot();
        time.sleep(0.1);
        dxl[i].EnableTorque()
        dxl[i].MoveTo(int(trg[i]), blocking=False)

  for i,_ in enumerate(DXL_ID):
    if mov[i]!=0.0:
      trg[i]= dxl[i].Position()+mov[i]
      print(i,c,mov,trg)
      dxl[i].MoveTo(int(trg[i]), blocking=False)
      #time.sleep(0.002)

  #time.sleep(0.002)
  #print('Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3}'.format(
    #[dxl[i].Position() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].Velocity() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].Current() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].PWM() for i,_ in enumerate(DXL_ID)]))
  print('Pos: {0}'.format(
    [dxl[i].Position() for i,_ in enumerate(DXL_ID)],))

is_running[0]= False
t1.join()

for i,_ in enumerate(DXL_ID):
  #dxl[i].PrintStatus()
  #dxl[i].PrintHardwareErrSt()
  dxl[i].DisableTorque()
  dxl[i].Quit()
