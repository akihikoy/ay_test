#!/usr/bin/python3
#\file    keyctrl.py
#\brief   Keyboard control a Robotiq gripper.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.26, 2017
import roslib
import rospy
from robotiq import TRobotiq

from kbhit2 import TKBHit
import threading
import sys

def ReadKeyboard(is_running, key_cmd):
  kbhit= TKBHit()
  while is_running[0] and not rospy.is_shutdown():
    c= kbhit.KBHit()
    key_cmd[0]= c
    rospy.sleep(0.0025)

if __name__ == '__main__':
  rospy.init_node('robotiq_test')
  rq= TRobotiq()
  rq.Init()

  key_cmd= [None]
  #using_threadkey= True
  using_threadkey= False
  if using_threadkey:
    is_running= [True]
    t1= threading.Thread(name='t1', target=lambda a1=is_running,a2=key_cmd: ReadKeyboard(a1,a2))
    t1.start()
  else:
    kbhit= TKBHit()

  trg= rq.status.gPO  #Current position

  while not rospy.is_shutdown():
    if using_threadkey:
      c= key_cmd[0]
    else:
      c= kbhit.KBHit()
      key_cmd[0]= c
    mov= 0.0
    if using_threadkey:  d= [2, 10]
    else:                d= [10, 100]
    #else:                d= [20, 255]
    if c is not None:
      if c=='q':  break
      elif c=='z':  mov= -d[1]
      elif c=='x':  mov= -d[0]
      elif c=='c':  mov= d[0]
      elif c=='v':  mov= d[1]

    if using_threadkey:
      if mov!=0:
        #trg= max(0,min(255,trg+mov))
        trg= max(0,min(255,rq.status.gPO+mov))
        print(c,mov,trg)
        #rq.MoveGripper(pos=int(trg), max_effort=255, speed=50, blocking=False)
        rq.MoveGripper(pos=int(trg), max_effort=100, speed=1, blocking=False)
        rospy.sleep(0.002)
      else:
        #rospy.sleep(0.0025)
        pass

    else:
      #trg= max(0,min(255,trg+mov))
      trg= max(0,min(255,rq.status.gPO+mov))
      print(c,mov,trg)
      #rq.MoveGripper(pos=int(trg), max_effort=255, speed=50, blocking=False)
      rq.MoveGripper(pos=int(trg), max_effort=100, speed=100, blocking=False)
      rospy.sleep(0.015)

    #else:
      #trg= max(0,min(255,rq.status.gPO+mov))
      #print c,mov,trg
      #rq.MoveGripper(pos=int(trg), max_effort=100, speed=10, blocking=False)
      #rospy.sleep(0.03)

    #rq.PrintStatus(rq.status)

  if using_threadkey:
    is_running[0]= False
    t1.join()

  print('done')


