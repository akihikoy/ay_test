#!/usr/bin/python3
#ref: http://stackoverflow.com/questions/14508906/sending-messages-between-class-threads-python
#http://ja.pymotw.com/2/Queue/
import threading
import queue
import time

Trigger= False  #global variable to communicate btwn the threads
QTrigger= queue.Queue()
IsActive= True

def Func1():
  global IsActive,Trigger
  while IsActive:
    if Trigger:
      Trigger= False
      print('p is pushed!!!')
    #time.sleep(0.001)  #Cpu usage is 100% if this is not used

def Func2():
  while IsActive:
    e= QTrigger.get()
    if e=='p':
      print('p is pushed!!!')
    elif e=='q':
      print('bye-bye!!!')

def MainThread():
  global IsActive
  while IsActive:
    line= input('q to quit, p to print > ')
    if line == 'q':
      IsActive= False
      QTrigger.put('q')
      break
    elif line == 'p':
      global Trigger
      Trigger= True
      QTrigger.put('p')
    else:
      print('  entered: ',line)

#t1= threading.Thread(name='func1', target=Func1)
t1= threading.Thread(name='func2', target=Func2)
t2= threading.Thread(name='main', target=MainThread)
#t1.setDaemon(True)
#t2.setDaemon(True)

t1.start()
t2.start()

t1.join()
t2.join()
print('Finished')

