#!/usr/bin/python3
#\file    lambda_local.py
#\brief   test lambda with local variable.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Feb.21, 2017
import time,threading

locker= threading.RLock()

def func(i, obj):
  while obj['count']>0:
    with locker:
      print('thread',i,obj,id(obj))
    obj['count']-= 1
    time.sleep(0.1)
  print('finished',i,obj,id(obj))

def func2(obj):
  func(obj['id'],obj)

#Two patterns
def make_thread1():
  threads= []
  for i in range(2):
    obj= {}
    obj['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=lambda: func(i,obj))
    t.start()
    threads.append(t)
  return threads

#Two patterns
def make_thread12():
  threads= []
  objs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['id']= i
    objs[i]['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=lambda: func2(objs[i]))
    t.start()
    threads.append(t)
  return threads,objs

#Two patterns
def make_thread13():
  threads= []
  objs= {}
  funcs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['id']= i
    objs[i]['count']= (i+2)*2
    funcs[i]= lambda: func2(objs[i])
    t= threading.Thread(name='func'+str(i), target=funcs[i])
    t.start()
    threads.append(t)
  return threads,objs,funcs

#Two patterns
def make_thread14():
  threads= []
  objs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['id']= i
    objs[i]['count']= (i+2)*2
    objs[i]['f']= lambda: func2(objs[i])
    t= threading.Thread(name='func'+str(i), target=objs[i]['f'])
    t.start()
    threads.append(t)
  return threads,objs

class TObj:
  def __init__(self):
    self.objs= None
  def f(self):
    func2(self.objs)

#One pattern (okay)
def make_thread15():
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

#Two patterns
def make_thread2():
  threads= []
  for i in range(2):
    obj= {}
    obj['count']= (i+2)*2
    t= threading.Thread(name='func'+str(i), target=lambda: func(i,obj))
    threads.append(t)
  print(threads, obj)
  for t in threads:
    t.start()
  return threads

#Two patterns
def make_thread3():
  threads= []
  funcs= []
  for i in range(2):
    obj= {}
    obj['count']= (i+2)*2
    funcs.append(lambda: func(i,obj))
  for i,f in enumerate(funcs):
    t= threading.Thread(name='func'+str(i), target=f)
    t.start()
    threads.append(t)
  #print threads, obj
  #for t in threads:
    #t.start()
  return threads

#Two patterns
def make_thread4():
  threads= []
  funcs= []
  objs= {}
  for i in range(2):
    objs[i]= {}
    objs[i]['count']= (i+2)*2
    #funcs.append(lambda: func(i,objs[i]))
  funcs= [lambda: func(0,objs[0]), lambda: func(1,objs[1])]
  print(funcs, objs)
  for i,f in enumerate(funcs):
    t= threading.Thread(name='func'+str(i), target=f)
    t.start()
    threads.append(t)
  #print threads, obj
  #for t in threads:
    #t.start()
  return threads,objs

if __name__=='__main__':
  #threads= make_thread1()
  #threads,objs= make_thread12()
  #with locker:  print threads,objs,id(objs[0]),id(objs[1])
  #threads,objs,funcs= make_thread13()
  #with locker:  print threads,(threads[0]._target),(threads[1]._target),objs,funcs,id(objs[0]),id(objs[1])
  #threads,objs= make_thread14()
  #with locker:  print threads,(threads[0]._target),(threads[1]._target),objs,id(objs[0]),id(objs[1])
  threads,classes= make_thread15()
  with locker:  print(threads,(threads[0]._target),(threads[1]._target),classes,classes[0].objs,classes[1].objs)
  #threads,objs= make_thread4()
  #with locker:  print objs
  for t in threads:
    t.join()

