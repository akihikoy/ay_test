#!/usr/bin/python3
#\file    singleton3.py
#\brief   Example of a singleton class
#         with deleting mechanism and reference counter.
#         with multi-singleton.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.27, 2018

import threading

class SingletonTest(object):
  #Multiple instances will be created for each self.x.
  _instance= {}
  _ref_counter= {}
  _lock= threading.Lock()
  def __new__(cls, *args, **kwargs):
    raise NotImplementedError('Do not initialize with the constructor.')
  @classmethod
  def new(cls, x, *args, **kwargs):
    if x not in cls._instance:
      with cls._lock:
        if x not in cls._instance:
          cls._instance[x]= super(SingletonTest,cls).__new__(cls)
          cls._instance[x].__init__(x, *args, **kwargs)
          cls._ref_counter[x]= 1
        else:  cls._ref_counter[x]+= 1
    else:  cls._ref_counter[x]+= 1
    return cls._instance[x]

  def delete(self):
    cls= self.__class__
    with cls._lock:
      if self.x not in cls._ref_counter or cls._ref_counter[self.x]==0:
        raise Exception('SingletonTest.delete is called with zombie.')
      cls._ref_counter[self.x]-= 1
      if cls._ref_counter[self.x]==0:
        print('Deleting',id(cls._instance[self.x]))
        instance= cls._instance[self.x]
        del cls._instance[self.x]
        del cls._ref_counter[self.x]

  def __del__(self):
    print('Called: __del__({s})'.format(s=id(self)))

  #NOTE: In a singleton, such an initializer is confusing since the execution timing is unclear.
  def __init__(self, x, y=3, z=-3):
    print('Called: __init__({s})'.format(s=id(self)))
    self.x= x
    self.y= y
    self.z= z
  def __repr__(self):
    return 'SingletonTest({0}, {1}, {2}, {3})'.format(id(self), self.x, self.y, self.z)
    #return 'SingletonTest(...)'


test_type= 2

if test_type==1:

  #test1= SingletonTest(101)  #WARNING: This calls SingletonTest.__new__
  test2= SingletonTest.new(102, z=30)
  print('test2=',test2)
  test2.delete()
  #del test2

  test3= SingletonTest.new(103, z=30)
  test6= SingletonTest.new(106)
  test7= SingletonTest.new(103)
  print('test3=',test3)
  print('test6=',test6)
  print('test7=',test7)
  test3.delete()
  test6.delete()
  test7.delete()


elif test_type==2:

  def F1():
    test4= SingletonTest.new(104, z=40)
    print('test4=',test4)
    test4.delete()

  test8= SingletonTest.new(104, z=40)

  F1()
  print('test8=',test8)
  test8.delete()

  F1()
  test5= SingletonTest.new(105)
  test9= SingletonTest.new(104)
  print('test5=',test5)
  print('test9=',test9)
  test5.delete()
  test9.delete()

