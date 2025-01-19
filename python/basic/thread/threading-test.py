#!/usr/bin/python3
#ref: http://ja.pymotw.com/2/threading/
import threading
import time

Counter= 0
IsActive= True

def func1():
  global IsActive
  while IsActive:
    line= input('q to quit, p to print > ')
    if line == 'q':
      IsActive= False
      break
    elif line == 'p':
      global Counter
      print('Counter=',Counter)
    else:
      print('  entered: ',line)

def func2():
  global IsActive,Counter
  while IsActive:
    time.sleep(0.5)
    Counter+= 1

t1= threading.Thread(name='func1', target=func1)
t2= threading.Thread(name='func2', target=func2)
#t1.setDaemon(True)
#t2.setDaemon(True)

t1.start()
t2.start()

print('t1.is_alive():',t1.is_alive())
print('t2.is_alive():',t2.is_alive())

t1.join()
t2.join()
print('Finished')
print('t1.is_alive():',t1.is_alive())
print('t2.is_alive():',t2.is_alive())

