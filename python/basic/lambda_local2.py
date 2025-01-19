#!/usr/bin/python3
#\file    lambda_local.py
#\brief   test lambda with local variable.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.21, 2017

'''This code is written for understanding a strange behavior of lambda. See
https://stackoverflow.com/questions/42380951/python-using-lambda-as-threading-target-causes-strange-behavior
'''

import time,threading

locker= threading.RLock()

def func(obj):
  while obj['count']>0:
    with locker: print('thread',obj,id(obj))
    obj['count']-= 1
    time.sleep(0.1)
  with locker: print('finished',obj,id(obj))

#Two patterns
def make_thread1():
  threads= []
  objs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['id']= i
    objs[i]['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=lambda: func(objs[i]))
    t.start()
    threads.append(t)
  return threads,objs

#
#cf. http://stackoverflow.com/questions/11087047/deferred-evaluation-with-lambda-in-python
def make_thread12():
  threads= []
  objs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['id']= i
    objs[i]['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=lambda i=i: func(objs[i]))
    t.start()
    threads.append(t)
  return threads,objs

class TObj:
  def __init__(self):
    self.objs= None
  def f(self):
    func(self.objs)

#One pattern (okay)
def make_thread2():
  threads= []
  classes= {}
  for i in range(2):
    classes[i]= TObj()
    classes[i].objs= {}
    classes[i].objs['id']= i
    classes[i].objs['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=classes[i].f)
    t.start()
    threads.append(t)
  return threads,classes

if __name__=='__main__':
  #threads,objs= make_thread1()
  threads,objs= make_thread12()
  #threads,classes= make_thread2()
  for t in threads:
    t.join()

