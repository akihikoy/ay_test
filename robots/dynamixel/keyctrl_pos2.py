#!/usr/bin/python
#Control dynamixel with key input (position control ver 2).

from dxl_util import *
from _config import *
import time

from kbhit2 import TKBHit
import threading
import sys

#Setup the device
dxl= TDynamixel1(DXL_TYPE,dev=DEV)
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
trg_offset= 0.0  #Virtual offset to increase position control power (like PI)
#Note: The effect is similar to increasing POS_I_GAIN but this offset is zero when the servo
#is moving; so it is more stable (less vibration/chatter).
sign= lambda x: 1 if x>0 else -1 if x<0 else 0

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= 0.0
  d= [10, 50]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','c','v'):  mov= {'z':-d[1],'x':-d[0],'c':d[0],'v':d[1]}[c]
    elif c in ('a','s','d','f'):
      #addr= 'PWM_LIMIT'
      addr= 'GOAL_PWM'
      max_pwm= min(max(dxl.Read(addr)+{'a':-50,'s':-5,'d':5,'f':50}[c], 0),dxl.MAX_PWM)
      #dxl.DisableTorque()
      dxl.Write(addr, max_pwm)
      #dxl.EnableTorque()
      dxl.CheckTxRxResult()
      print addr,':',max_pwm,dxl.Read(addr)
    elif c=='r':
      dxl.Reboot();
      time.sleep(0.1);
      dxl.EnableTorque()
      dxl.MoveTo(int(trg), blocking=False)

  if mov!=0:
    #trg= max(0,min(255,trg+mov))
    #trg= max(0,min(255,dxl.Position()+mov))
    trg= dxl.Position()+mov
    print c,mov,trg
  else:
    #time.sleep(0.0025)
    pass

  th1= 3
  th2= 3
  ostep= 3
  #max_pwm= dxl.MAX_PWM*0.9
  max_pwm= dxl.Read('GOAL_PWM')*0.9
  pos,vel,pwm= dxl.Position(), dxl.Velocity(), dxl.PWM()
  if trg_offset!=0 and sign(trg-pos)!=sign(trg_offset):
    trg_offset= 0.0
  #elif trg_offset!=0 and abs(trg-pos)>th1 and abs(vel)>th2:
    ##trg_offset= trg_offset - ostep*sign(trg-pos)
    #trg_offset= 0.0
  elif abs(vel)>=th2:
    trg_offset= 0.0
  elif abs(trg-pos)>th1 and abs(vel)<th2 and abs(pwm)<max_pwm:
    trg_offset= trg_offset + ostep*sign(trg-pos)
  #print trg-pos, trg_offset
  dxl.MoveTo(int(trg+trg_offset), blocking=False)
  time.sleep(0.002)
  #print 'Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3} \t TEMP: {4}'.format(dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM(),dxl.Temperature())
  print 'Err: {5} \t offset: {6} \t P: {0} \t V: {1} \t C: {2} \t PWM: {3} \t TEMP: {4}'.format(
    dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM(),dxl.Temperature(),
    trg-pos, trg_offset)

is_running[0]= False
t1.join()


dxl.PrintStatus()
dxl.PrintHardwareErrSt()
#dxl.DisableTorque()
dxl.Quit()
