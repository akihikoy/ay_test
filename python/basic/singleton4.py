#!/usr/bin/python
#\file    singleton4.py
#\brief   Example of a singleton class
#         with deleting mechanism and reference counter.
#         with multi-singleton.
#         implemented with the metaclass.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.27, 2018
'''
Refs.
https://kiwamiden.com/singleton-in-python
'''

import threading

'''
Meta-class to generate a singleton class with multiple instances.
An instance is created for each key variable x.
NOTE: This is EXPERIMENTAL.  Especially not tested with threading.
Usage:
  Define a class where this class is used as the meta class.

  class X(object):
    __metaclass__= TMultiSingleton

  In X.__init__, the first argument should be the key value (x).
  Use X.Delete(x) to delete the instance (or reduce the reference count).
  Use X.NumReferences(x) to get the number of references (==reference count).
  X can be instantiated as a usual class:
    x1,x2,x3= X(1),X(1),X(2)
  where x1 and x2 are the same instance (ids are the same) since they have the same key value (1).
  x3 is a different instance.
'''
class TMultiSingleton(type):
  _instance= None
  _ref_counter= None
  _lock= None

  def __call__(cls, x, *args, **kwargs):
    if cls._instance is None:
      cls._instance= {}
      cls._ref_counter= {}
      cls._lock= threading.Lock()
    if x not in cls._instance:
      with cls._lock:
        if x not in cls._instance:
          print 'Creating',cls.__name__,x
          cls._instance[x]= super(TMultiSingleton,cls).__call__(x, *args, **kwargs)
          cls._ref_counter[x]= 1
        else:  cls._ref_counter[x]+= 1
    else:  cls._ref_counter[x]+= 1
    return cls._instance[x]

  def Delete(self, x):
    cls= self
    with cls._lock:
      if x not in cls._ref_counter or cls._ref_counter[x]==0:
        raise Exception('TMultiSingleton.Delete is called with zombie.')
      cls._ref_counter[x]-= 1
      if cls._ref_counter[x]==0:
        print 'Deleting',cls.__name__,x
        instance= cls._instance[x]
        del cls._instance[x]
        del cls._ref_counter[x]

  def NumReferences(self, x):
    cls= self
    if cls._ref_counter is None or x not in cls._ref_counter:  return 0
    return cls._ref_counter[x]


class SingletonTest(object):
  __metaclass__= TMultiSingleton

  def delete(self):
    self.__class__.Delete(self.x)

  def __del__(self):
    print 'Called: __del__({s})'.format(s=id(self))

  #NOTE: In a singleton, such an initializer is confusing since the execution timing is unclear.
  def __init__(self, x, y=3, z=-3):
    print 'Called: __init__({s})'.format(s=id(self))
    self.x= x
    self.y= y
    self.z= z
  def __repr__(self):
    return 'SingletonTest({0}, {1}, {2}, {3})'.format(id(self), self.x, self.y, self.z)
    #return 'SingletonTest(...)'

class SingletonTest2(object):
  __metaclass__= TMultiSingleton

  def delete(self):
    self.__class__.Delete(self.x)

  def __del__(self):
    print 'Called: __del__({s})'.format(s=id(self))

  #NOTE: In a singleton, such an initializer is confusing since the execution timing is unclear.
  def __init__(self, x, y=3, z=-3):
    print 'Called: __init__({s})'.format(s=id(self))
    self.x= x
    self.y= y
    self.z= z
  def __repr__(self):
    return 'SingletonTest2({0}, {1}, {2}, {3})'.format(id(self), self.x, self.y, self.z)
    #return 'SingletonTest2(...)'


test_type= 1

if test_type==1:

  test1= SingletonTest(101)
  test2= SingletonTest(102, z=30)
  print 'test1=',test1
  print 'test2=',test2
  test2.delete()
  #del test2

  test3= SingletonTest(103, z=30)
  test6= SingletonTest(106)
  test7= SingletonTest(103)
  print 'test3=',test3
  print 'test6=',test6
  print 'test7=',test7
  print 'NumReferences(103)=',SingletonTest.NumReferences(103)
  test3.delete()
  print 'NumReferences(103)=',SingletonTest.NumReferences(103)
  test6.delete()
  print 'NumReferences(103)=',SingletonTest.NumReferences(103)
  test7.delete()
  print 'NumReferences(103)=',SingletonTest.NumReferences(103)


elif test_type==2:

  def F1():
    test4= SingletonTest(104, z=40)
    print 'test4=',test4
    test4.delete()

  test8= SingletonTest(104, z=40)

  F1()
  print 'test8=',test8
  test8.delete()

  F1()
  test5= SingletonTest(105)
  test9= SingletonTest(104)
  print 'test5=',test5
  print 'test9=',test9
  test5.delete()
  test9.delete()

elif test_type==3:

  test1= SingletonTest(101)
  test2= SingletonTest(101, z=30)
  test3= SingletonTest2(101, z=60)
  test4= SingletonTest2(101)
  print 'test1=',test1
  print 'test2=',test2
  print 'test3=',test3
  print 'test4=',test4
  test1.delete()
  test2.delete()
  test3.delete()
  test4.delete()

