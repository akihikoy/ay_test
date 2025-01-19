#!/usr/bin/python3
#ref: http://ja.pymotw.com/2/threading/
import threading
import time
import sys

class TContainer: pass
l= TContainer()

l.Counter1= 0
l.Counter2= 100
l.IsActive= True
l.Locker1= threading.RLock()
l.Locker2= threading.RLock()

def func1():
  while l.IsActive:
    line= input('q to quit, p to print > ')
    if line == 'q':
      l.IsActive= False
      break
    elif line == 'p':
      with l.Locker2:
        print('Counter2=',l.Counter2)
      with l.Locker1:
        print('Counter1=', end=' ')
        for i in range(10):
          print(l.Counter1, end=' ')
          sys.stdout.flush()
          time.sleep(0.1)
        print('')
      with l.Locker2:
        print('Counter2=',l.Counter2)
    else:
      print('  entered: ',line)

def func2():
  while l.IsActive:
    time.sleep(0.5)
    with l.Locker1:
      l.Counter1+= 1

def func3():
  while l.IsActive:
    time.sleep(0.1)
    #TEST:A
    #with l.Locker2:
      #l.Counter2-= 1
    #TEST:B
    #if l.Locker2.acquire(False):
      #l.Counter2-= 1
      #l.Locker2.release()
    #else:
      #print l.Locker2
    #TEST:C
    #print l.Locker2,l.Locker2._is_owned(),l.Locker2._RLock__count
    #if not l.Locker2._is_owned():  #Do not use this (in Py2).
    #if l.Locker2._RLock__count==0:  #Is not available in Py3.
    if not l.Locker2._is_owned():  #Use this in Py3 as it is improved.
      #print '#',l.Locker2,l.Locker2._is_owned(),l.Locker2._RLock__count
      with l.Locker2:
        l.Counter2-= 1
    else:
      print(l.Locker2)

t1= threading.Thread(name='func1', target=func1)
t2= threading.Thread(name='func2', target=func2)
t3= threading.Thread(name='func3', target=func3)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()
print('Finished')

