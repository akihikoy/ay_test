#!/usr/bin/python3
#Control dynamixel with key input (position control).
#NOTE: Run before this script: ../fix_usb_latency.sh

'''
Keyboard interface:
  'q':  Quit.
  'z','x','c','v': Close large, close little, open little, open large.
  'a','s','d': Move to preset[0], [1], [2]
'''
preset= [520,260,0]

from dxl_util import *
from _config import *
import time

from kbhit2 import TKBHit
import threading
import sys

#Setup the device
dxl= [TDynamixel1(DXL_TYPE) for _ in DXL_ID]
for i,_ in enumerate(DXL_ID):
  dxl[i].Id= DXL_ID[i]
  dxl[i].Baudrate= BAUDRATE
  dxl[i].Setup()
  dxl[i].EnableTorque()

#Move to the current position
p_start= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]
for i,_ in enumerate(DXL_ID):
  dxl[i].MoveTo(p_start[i])
time.sleep(0.5)  #wait .5 sec
print('Current position=',[dxl[i].Position() for i,_ in enumerate(DXL_ID)])


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

trg= [dxl[i].Position() for i,_ in enumerate(DXL_ID)]  #Current position

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= 0.0
  d= [20, 50]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','c','v'):  mov= {'z':-d[1],'x':-d[0],'c':d[0],'v':d[1]}[c]
    elif c in ('a','s','d'):
      dp= {'a':preset[0],'s':preset[1],'d':preset[2]}[c]
      trg= [2048+dp,2048-dp,2048+dp,2048-dp]
      for i,_ in enumerate(DXL_ID):
        dxl[i].MoveTo(int(trg[i]), blocking=False)
    elif c=='r':
      for i,_ in enumerate(DXL_ID):
        dxl[i].Reboot();
        time.sleep(0.1);
        dxl[i].EnableTorque()
        dxl[i].MoveTo(int(trg[i]), blocking=False)

  if mov!=0:
    trg= [dxl[0].Position()+mov, dxl[1].Position()+mov, dxl[2].Position()+mov, dxl[3].Position()+mov]
    print(c,mov,trg)
    for i,_ in enumerate(DXL_ID):
      dxl[i].MoveTo(int(trg[i]), blocking=False)
    #time.sleep(0.002)
  else:
    #time.sleep(0.0025)
    pass

  #time.sleep(0.002)
  #print 'Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3} \t TEMP: {4}'.format(
    #[dxl[i].Position() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].Velocity() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].Current() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].PWM() for i,_ in enumerate(DXL_ID)],
    #[dxl[i].Temperature() for i,_ in enumerate(DXL_ID)])
  print('Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3}'.format(
    [dxl[i].Position() for i,_ in enumerate(DXL_ID)],
    [dxl[i].Velocity() for i,_ in enumerate(DXL_ID)],
    [dxl[i].Current() for i,_ in enumerate(DXL_ID)],
    [dxl[i].PWM() for i,_ in enumerate(DXL_ID)]))

is_running[0]= False
t1.join()


for i,_ in enumerate(DXL_ID):
  dxl[i].PrintStatus()
  dxl[i].PrintHardwareErrSt()
  #dxl[i].DisableTorque()
  dxl[i].Quit()
