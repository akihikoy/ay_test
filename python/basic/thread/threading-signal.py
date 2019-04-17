#!/usr/bin/python
'''
This is what I want to do, but it does not work!
'''

import threading
import Queue

queue= Queue.Queue()

def Func1():
  while True:
    data= queue.get()
    print 'Func1:got',data
    if data=='q':  break

def Func2():
  while True:
    data= queue.get()
    print 'Func2:got',data
    if data=='q':  break

def MainThread():
  while True:
    data= raw_input('q to quit > ')
    queue.put(data)
    if data=='q':  break

t1= threading.Thread(name='func1', target=Func1)
t2= threading.Thread(name='func2', target=Func2)
tm= threading.Thread(name='main', target=MainThread)
t1.start()
t2.start()
tm.start()

t1.join()
t2.join()
tm.join()

