#!/usr/bin/python
#Control dynamixel with key input (position control).
#NOTE: Run before this script: ../fix_usb_latency.sh

'''
Keyboard interface:
  'q':  Quit.
  'z','x','c','v': Close large, close little, open little, open large.
  'a','s','d': Move to preset
'''

from dxl_fd2f4dof import *
import time

from kbhit2 import TKBHit
import threading
import sys
import numpy as np

#Setup the device
gripper= TFD2F4DoF()
gripper.Setup()
gripper.EnableTorque()

#Move to the current position
pose= gripper.Position()
gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),pose)})
time.sleep(0.5)  #wait .5 sec
print 'Current position=',gripper.Position()


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

trg= gripper.Position()  #Current position

try:
  while True:
    with key_locker:
      c= key_cmd[0]; key_cmd[0]= None
    mov= None
    #d= 0.01
    d= 0.05
    if c is not None:
      if c=='q':  break
      elif c in ('z','x'):  mov= {'z':[-d,-d,0,0],'x':[d,d,0,0]}[c]
      elif c in ('c','v'):  mov= {'c':[0,0,-d,-d],'v':[0,0,d,d]}[c]
      elif c in ('a','s','d'):
        trg= np.array([0.28, 0.28, 0.72, 0.72]) + {'a':[0,0,0,0],'s':[0,0,-0.6,-0.6],'d':[-0.25,-0.25,0,0]}[c]
        #NOTE:MODEL2023
        #trg= np.array([-0.5, -0.5, 1.0, 1.0]) + {'a':[0,0,0,0],'s':[0,0,-0.6,-0.6],'d':[-0.2,-0.2,0,0]}[c]
        gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),trg)})
      elif c=='r':
        gripper.Reboot()
        gripper.EnableTorque()
        trg= gripper.Position()

    if mov is not None:
      trg= np.array(gripper.Position()) + mov
      print c,mov,trg
      gripper.MoveTo({jname:p for jname,p in zip(gripper.JointNames(),trg)})
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
    #print 'Pos: {0} \t Vel: {1} \t Curr: {2} \t PWM: {3}'.format(
      #[dxl[i].Position() for i,_ in enumerate(DXL_ID)],
      #[dxl[i].Velocity() for i,_ in enumerate(DXL_ID)],
      #[dxl[i].Current() for i,_ in enumerate(DXL_ID)],
      #[dxl[i].PWM() for i,_ in enumerate(DXL_ID)])
    print 'Position=','[',', '.join(['{:.4f}'.format(p) for p in gripper.Position()]),']'
finally:
  print 'Finishing...'
  is_running[0]= False
  t1.join()

#gripper.PrintStatus()
#gripper.DisableTorque()
gripper.Quit()
