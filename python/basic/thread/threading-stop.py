#!/usr/bin/python
#ref: http://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
import threading
import time

Trigger= False  #global variable to communicate btwn the threads

def Func1():
  global Trigger
  while True:
    if Trigger:
      Trigger= False
      print 'p is pushed!!!'
    time.sleep(0.001)  #Cpu usage is 100% if this is not used

def MainThread():
  while True:
    line= raw_input('q to quit, p to print > ')
    if line == 'q':
      break
    elif line == 'p':
      global Trigger
      Trigger= True
    else:
      print '  entered: ',line

t1= threading.Thread(name='func1', target=Func1)
t2= threading.Thread(name='main', target=MainThread)
#t1.setDaemon(True)
#t2.setDaemon(True)

t1.start()
t2.start()

t2.join()
t1._Thread__stop()  #FORCE TO STOP THE THREAD.
                    #WARNING: _Thread__stop does not work sometimes as written in the ref.


t1.join()
print 'Finished'

