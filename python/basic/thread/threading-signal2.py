#!/usr/bin/python
#see http://stackoverflow.com/questions/27561654/python-queue-get-from-multiple-threads-or-signal

import threading
import Queue

#class TSignal(Queue.Queue):
  #def __init__(self):
    #Queue.Queue.__init__(self)
    #self.counter= 0
    #self.locker1= threading.Lock()
    #self.locker2= threading.Lock()
    #self.data= None
  #def get(self):
    #self.locker1.acquire()
    #if self.counter==0:
      #with self.locker2:
        #self.counter+= 1
        #self.locker1.release()
        #self.data= Queue.Queue.get(self)
        #self.counter-= 1
        #return self.data
    #else:
      #self.counter+= 1
      #self.locker1.release()
      #with self.locker2:
        #self.counter-= 1
        #return self.data

#class TSignal(Queue.Queue):
  #def __init__(self):
    #Queue.Queue.__init__(self)
    #self.listener= 0
    #self.time= 0
    #self.table= {}  #idx-->time
    #self.data= {}  #time-->data
    #self.locker_time= threading.Lock()
    #self.locker_get= threading.Lock()
    #self.locker_getid= threading.Lock()
  #def get(self, idx):
    #with self.locker_time:
      #old_data= self.table[idx]<self.time
    #if old_data:
      #self.table[idx]+= 1
      #return self.data[self.table[idx]]
    #else:
      #with self.locker_get:
        #d= Queue.Queue.get(self)
        #with self.locker_time:
          #self.time+= 1
          #self.data[self.time]= d
        #self.table[idx]= self.time
        #return d
  #def getid(self):
    #with self.locker_getid:
      #idx= self.listener
      #self.listener+= 1
      #self.table[idx]= self.time
      #return idx


'''TSignal class for a thread to send a message to several threads.
This is an extension of Queue.Queue.  The idea is queue-per-thread.
Usage:
  In a wider scope, define this object, like a queue.
    signal= TSignal()
  In receivers, you can write either a with-statement form or a normal form.
    with signal.NewQueue() as queue:
      #use queue normally; e.g. data= queue.get()
  Or:
    queue= signal.NewQueue()
    #use queue normally; e.g. data= queue.get()
    #at the end of this scope, queue is automatically released,
    #but if you want to do it explicitly, you can use either:
    del queue
    #or
    queue= None
  In sender(s), you can write normally:
    signal.put(data)
'''
class TSignal:
  def __init__(self):
    self.queues= {}  #Map from index to queue
    self.counter= 0
    self.locker= threading.Lock()
  def NewQueue(self):
    idx= self.counter
    self.counter+= 1
    with self.locker:
      self.queues[idx]= Queue.Queue()
    queue= self.TQueue(self,idx,self.queues[idx])
    return queue
  def DeleteQueue(self,idx):
    with self.locker:
      if idx in self.queues:
        del self.queues[idx]
  def put(self,item,block=True,timeout=None):
    with self.locker:
      items= self.queues.items()
    for idx,queue in items:
      queue.put(item,block,timeout)
  class TQueue:
    def __init__(self,parent,idx,queue):
      self.parent= parent
      self.idx= idx
      self.queue= queue
    def __del__(self):
      self.parent.DeleteQueue(self.idx)
    def __enter__(self):
      return self
    def __exit__(self,e_type,e_value,e_traceback):
      self.parent.DeleteQueue(self.idx)
    def get(self,block=True,timeout=None):
      return self.queue.get(block,timeout)

signal= TSignal()

#Usage 1: write as a with statement
def Func1():
  with signal.NewQueue() as queue:
    while True:
      data= queue.get()
      print '\nFunc1:got[%r]\n'%data
      if data=='q':  break

#Usage 1: define a local variable
def Func2():
  queue= signal.NewQueue()
  while True:
    data= queue.get()
    print '\nFunc2:got[%r]\n'%data
    if data=='q':  break
  #NOTE: queue is automatically freed after this function, but if want to do it now,
  #you can use either: [del queue] or [queue= None]

def MainThread():
  while True:
    data= raw_input('q to quit > ')
    signal.put(data)
    if data=='q':  break

t1= threading.Thread(name='func1', target=Func1)
t2= threading.Thread(name='func2', target=Func2)
tm= threading.Thread(name='main', target=MainThread)
t1.start()
t2.start()
tm.start()

print 'signal.queues:',signal.queues

t1.join()
t2.join()
tm.join()

print 'signal.queues:',signal.queues

