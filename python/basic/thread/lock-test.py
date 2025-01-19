#!/usr/bin/python3
#\file    lock-test.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.27, 2020

import time
import threading

print_locker= threading.RLock()
#print_locker= threading.Lock()  #The deadlock does not happen with Lock.

IsActive= True
def func1():
  global IsActive,print_locker
  while IsActive:
    print('P5')#,print_locker,threading.enumerate()
    with print_locker:  #Deadlock occasionally happens here-1
      print('func1',time.time())
    #time.sleep(0.0001)
  print('P6',print_locker,threading.enumerate())

try:
  t1= threading.Thread(name='func1', target=func1)
  t1.start()
  while True:
    with print_locker:
      print('main',time.time())
    time.sleep(0.001)
except KeyboardInterrupt:
  IsActive= False
  print('P4')#,print_locker,threading.enumerate()
  t1.join()  #Deadlock occasionally happens here-1
  print('Terminated')#,time.time(),threading.enumerate()

print('P1',print_locker,threading.enumerate())
with print_locker:  #Deadlock occasionally happens here-2
  print('P2',print_locker,threading.enumerate())

print('P3',print_locker,threading.enumerate())
