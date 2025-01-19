#!/usr/bin/python3
#ref: http://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
import threading
import time

Trigger= False  #global variable to communicate btwn the threads
Running= False

def Func1():
  global Trigger, Running
  while Running:
    if Trigger:
      Trigger= False
      print('p is pushed!!!')
    time.sleep(0.001)  #Cpu usage is 100% if this is not used

def MainThread():
  global Trigger, Running
  while Running:
    line= input('q to quit, p to print > ')
    if line == 'q':
      Running= False
      break
    elif line == 'p':
      Trigger= True
    else:
      print('  entered: ',line)

t1= threading.Thread(name='func1', target=Func1)
t2= threading.Thread(name='main', target=MainThread)
#t1.setDaemon(True)
#t2.setDaemon(True)

Running= True
t1.start()
t2.start()

t2.join()

'''
t1._Thread__stop()  #FORCE TO STOP THE THREAD.
                    #WARNING: _Thread__stop does not work sometimes as written in the ref.
WARNING: _Thread__stop is not a regular way to stop a thread,
and in Python3, it is no longer available.
'''

t1.join()
print('Finished')

