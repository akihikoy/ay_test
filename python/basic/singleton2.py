#!/usr/bin/python3
#\file    singleton2.py
#\brief   Example of a singleton class
#         with deleting mechanism and reference counter.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.27, 2018

import threading

class SingletonTest(object):
  _instance= None
  _ref_counter= 0
  _lock= threading.Lock()
  def __new__(cls, *args, **kwargs):
    raise NotImplementedError('Do not initialize with the constructor.')
  @classmethod
  def new(cls, *args, **kwargs):
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance= super(SingletonTest,cls).__new__(cls)
          cls._instance.__init__(*args, **kwargs)
          cls._ref_counter= 1
        else:  cls._ref_counter+= 1
    else:  cls._ref_counter+= 1
    return cls._instance

  def delete(self):
    cls= self.__class__
    with cls._lock:
      if cls._ref_counter==0:  raise Exception('SingletonTest.delete is called with zombie.')
      cls._ref_counter-= 1
      if cls._ref_counter==0:
        print('Deleting',id(cls._instance))
        instance= cls._instance
        cls._instance= None
        del instance

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


test_type= 1

if test_type==1:

  #test1= SingletonTest(101)  #WARNING: This calls SingletonTest.__new__
  test2= SingletonTest.new(102, z=30)
  print('test2=',test2)
  test2.delete()
  #del test2

  test3= SingletonTest.new(103)
  test6= SingletonTest.new(106)
  #print 'test2=',test2
  print('test3=',test3)
  print('test6=',test6)
  test3.delete()
  test6.delete()


elif test_type==2:

  def F1():
    test4= SingletonTest.new(104, z=40)
    print('test4=',test4)
    test4.delete()

  F1()

  test5= SingletonTest.new(105)
  print('test5=',test5)
  test5.delete()

