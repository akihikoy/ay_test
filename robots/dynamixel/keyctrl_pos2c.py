#!/usr/bin/python
#Control dynamixel with key input (position control ver 2b).
#The same as ver 2 except for the implementation (using thread).

from dxl_util import *
from _config import *
import time

from kbhit2 import TKBHit
from rate_adjust import TRate
import threading
import sys

#Setup the device
dxl= TDynamixel1(DXL_TYPE)
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


#------------------------------------------------------------------------------

'''
Holding controller for Dynamixel.
It uses a virtual offset to increase position control power (like PI).
The effect is similar to increasing POS_I_GAIN but this offset is zero when the servo
is moving; it is more stable (i.e. less vibration/chatter).

Example:
  port_locker= threading.RLock()
  def holding_observer():
    with port_locker:
      pos,vel,pwm= dxl.Position(), dxl.Velocity(), dxl.PWM()
    return pos,vel,pwm

  def holding_controller(target_position):
    with port_locker:
      dxl.MoveTo(target_position, blocking=False)

  holding= TDxlHolding()
  holding.observer= holding_observer
  holding.controller= holding_controller
  holding.SetTarget(TARGET_POS, MAX_PWM)
  holding.Start()

  user-defined-loop:
    ...
    holding.SetTarget(TARGET_POS, MAX_PWM)
    ...

  holding.Stop()

'''
class TDxlHolding(object):
  def __init__(self, rate=30):
    self.trg_pos= 2048
    self.max_pwm= 100
    self.is_running= False

    self.th_p= 3
    self.th_v= 3
    self.ostep= 3
    self.ctrl_rate= rate

    self.observer= None  #Should be: pos,vel,pwm= observer()
    self.controller= None  #Should be: controller(target_position)

  #Set target position.
  #  trg_pos: Target position.
  #  max_pwm: Maximum effort; when pwm exceeds this value, we don't increase the offset.
  def SetTarget(self, trg_pos, max_pwm=None):
    self.trg_pos= trg_pos
    self.max_pwm= max_pwm if max_pwm is not None else self.max_pwm

  def Start(self):
    self.Stop()
    self.thread= threading.Thread(name='holding', target=self.Loop)
    self.is_running= True
    self.thread.start()

  def Stop(self):
    if self.is_running:
      self.is_running= False
      self.thread.join()

  def Loop(self):
    sign= lambda x: 1 if x>0 else -1 if x<0 else 0

    rate= TRate(self.ctrl_rate)

    #Virtual offset:
    self.trg_offset= 0.0

    while self.is_running:
      pos,vel,pwm= self.observer()

      if self.trg_offset!=0 and sign(self.trg_pos-pos)!=sign(self.trg_offset):
        self.trg_pos= self.trg_pos+self.trg_offset
        self.trg_offset= 0.0
      elif abs(vel)>=self.th_v:
        self.trg_pos= self.trg_pos+self.trg_offset
        self.trg_offset= 0.0
      elif abs(self.trg_pos-pos)>self.th_p and abs(vel)<self.th_v and abs(pwm)<self.max_pwm:
        self.trg_offset= self.trg_offset + self.ostep*sign(self.trg_pos-pos)

      self.controller(int(self.trg_pos+self.trg_offset))

      rate.sleep()

#------------------------------------------------------------------------------

port_locker= threading.RLock()
def holding_observer():
  with port_locker:
    pos,vel,pwm= dxl.Position(), dxl.Velocity(), dxl.PWM()
  return pos,vel,pwm

def holding_controller(target_position):
  with port_locker:
    dxl.MoveTo(target_position, blocking=False)

holding= TDxlHolding()
holding.observer= holding_observer
holding.controller= holding_controller
holding.SetTarget(dxl.Position(), dxl.Read('GOAL_PWM')*0.9)
holding.Start()

while True:
  with key_locker:
    c= key_cmd[0]; key_cmd[0]= None
  mov= 0.0
  d= [10, 50]
  if c is not None:
    if c=='q':  break
    elif c in ('z','x','c','v'):  mov= {'z':-d[1],'x':-d[0],'c':d[0],'v':d[1]}[c]
    elif c in ('a','s','d','f'):
      with port_locker:
        #addr= 'PWM_LIMIT'
        addr= 'GOAL_PWM'
        max_pwm= min(max(dxl.Read(addr)+{'a':-50,'s':-5,'d':5,'f':50}[c], 0),dxl.MAX_PWM)
        #dxl.DisableTorque()
        dxl.Write(addr, max_pwm)
        #dxl.EnableTorque()
        dxl.CheckTxRxResult()
        print addr,':',max_pwm,dxl.Read(addr)
    elif c=='r':
      with port_locker:
        dxl.Reboot();
        time.sleep(0.1);
        dxl.EnableTorque()
        dxl.MoveTo(int(trg), blocking=False)

  if mov!=0:
    #trg= max(0,min(255,trg+mov))
    #trg= max(0,min(255,dxl.Position()+mov))
    with port_locker:
      trg= dxl.Position()+mov
      max_pwm= dxl.Read('GOAL_PWM')*0.9
    print c,mov,trg
    #dxl.MoveTo(int(trg+trg_offset), blocking=False)
    #dxl.MoveTo(int(trg), blocking=False)
    holding.SetTarget(trg, max_pwm)
  else:
    #time.sleep(0.0025)
    pass

  time.sleep(0.02)
  #print 'Err: {5} \t offset: {6} \t P: {0} \t V: {1} \t C: {2} \t PWM: {3} \t TEMP: {4}'.format(
    #dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM(),dxl.Temperature(),
    #trg-pos, trg_offset)
  #with port_locker:
    #print 'Err: {5} \t offset: {6} \t P: {0} \t V: {1} \t C: {2} \t PWM: {3} \t TEMP: {4}'.format(
      #dxl.Position(),dxl.Velocity(),dxl.Current(),dxl.PWM(),dxl.Temperature(),
      #holding.trg-dxl.Position(), holding.trg_offset)

is_running[0]= False
t1.join()
holding.Stop()

dxl.PrintStatus()
dxl.PrintHardwareErrSt()
#dxl.DisableTorque()
dxl.Quit()
