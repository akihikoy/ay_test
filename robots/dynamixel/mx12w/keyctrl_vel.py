#!/usr/bin/python
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
dxl= [TDynamixel1(DXL_TYPE[i],dev=DXL_DEV) for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].OpMode= OP_MODE
  dxl[i].Setup()
  dxl[i].EnableTorque()

#Move to the current position
p_start= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].MoveTo(p_start[i])
time.sleep(0.5)  #wait .5 sec
print 'Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)]


def ReadKeyboard(is_running, key_cmd, key_locker):
  kbhit= TKBHit()
  dt_hold= 0.1
  t_prev= 0
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

trg= [0]

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= [0.0]
  d= [100, 500]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','c','v'):  mov[0]= {'z':-d[1],'x':-d[0],'c':d[0],'v':d[1]}[c]
    elif c=='r':
      for i,_ in enumerate(DXL_ID):
        dxl[i].Reboot();
        time.sleep(0.1);
        dxl[i].EnableTorque()
        #dxl[i].MoveTo(int(trg[i]), blocking=False)

  if mov[0]!=0.0:
    trg[0]= trg[0]+mov[0]
    print c,mov,trg
  dxl[0].SetVelocity(int(trg[0]))

  #time.sleep(0.002)
  print 'Pos: {0} \t Vel: {1}'.format(
    [dxl[i].Position() for i,_ in enumerate(DXL_ID)],
    [dxl[i].Velocity() for i,_ in enumerate(DXL_ID)])

is_running[0]= False
t1.join()

for i,_ in enumerate(DXL_ID):
  dxl[i].PrintStatus()
  #dxl[i].PrintHardwareErrSt()
  dxl[i].DisableTorque()
  dxl[i].Quit()
