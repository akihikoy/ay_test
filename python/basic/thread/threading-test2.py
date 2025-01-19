#!/usr/bin/python3
#ref: http://ja.pymotw.com/2/threading/
import threading
import time

class Test1:
  def __init__(self):
    self.counter= 0
    self.is_active= True

  def __del__(self):
    self.t1.join()
    self.t2.join()
    print('Finished')

  def Start(self):
    self.t1= threading.Thread(name='func1', target=self.Func1)
    self.t2= threading.Thread(name='func2', target=self.Func2)
    #self.t1.setDaemon(True)
    #self.t2.setDaemon(True)

    self.t1.start()
    self.t2.start()

  def Join(self):
    self.t1.join()
    self.t2.join()

  def Func1(self):
    while self.is_active:
      line= input('q to quit, p to print > ')
      if line == 'q':
        self.is_active= False
        break
      elif line == 'p':
        print('counter=',self.counter)
      else:
        print('  entered: ',line)

  def Func2(self):
    while self.is_active:
      time.sleep(0.5)
      self.counter+= 1


test1= Test1()
test1.Start()
test1.Join()

