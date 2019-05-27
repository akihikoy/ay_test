#!/usr/bin/python
#Control dynamixel with key input (current-based position control).

from dxl_util import *
from _config import *
import time

from kbhit2 import TKBHit
import threading
import sys

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
dxl.OpMode= 'CURRPOS'
dxl.Id= DXL_ID
dxl.Baudrate= BAUDRATE
dxl.Setup()
dxl.EnableTorque()

#Move to initial position
p_start= 2100
dxl.MoveTo(p_start)
time.sleep(0.5)  #wait .5 sec
print 'Current position=',dxl.Position()


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

trg= dxl.Position()  #Current position

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= 0.0
  d= [10, 50]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','c','v'):  mov= {'z':-d[1],'x':-d[0],'c':d[0],'v':d[1]}[c]
    elif c=='r':
      dxl.Reboot();
      time.sleep(0.1);
      dxl.EnableTorque()
      dxl.MoveToC(int(trg), current=dxl.CurrentLimit, blocking=False)

  if mov!=0:
    #trg= max(0,min(255,trg+mov))
    #trg= max(0,min(255,dxl.Position()+mov))
    trg= dxl.Position()+mov
    #trg= trg+mov
    print c,mov,trg
    #dxl.MoveTo(int(trg), blocking=False)
    curr= dxl.CurrentLimit if mov>0 else -dxl.CurrentLimit
    #curr= 50*(1 if mov>0 else -1)
    dxl.MoveToC(int(trg), current=curr, blocking=False)
    time.sleep(0.002)
    print 'Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3} \t TEMP: {4}'.format(dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM(),dxl.Temperature())
  else:
    #time.sleep(0.0025)
    pass

is_running[0]= False
t1.join()


dxl.PrintStatus()
dxl.PrintHardwareErrSt()
#dxl.DisableTorque()
dxl.Quit()
