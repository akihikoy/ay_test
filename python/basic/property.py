#!/usr/bin/python3
#\file    property.py
#\brief   Test of @property.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.20, 2015

'''
Refs.
http://qiita.com/knzm/items/a8a0fead6e1706663c22
http://stackoverflow.com/questions/15458613/python-why-is-read-only-property-writable

NOTE:
- Each class that defines property should be a sub class of 'object' class.
'''

class TTest1(object):
  def __init__(self):
    self.x= 2

  @property
  def square(self):
    return self.x**2

class TTest2(object):
  def __init__(self):
    self.x= 3

  @property
  def square(self):
    return self.x**2

  @square.setter
  def square(self, value):
    raise Exception('not writable')

class TTest3(object):
  def __init__(self):
    self.x= 4

  @property
  def square(self):
    return self.x**2

  @square.getter
  def square(self):
    return self.x**3    #WARNING: This is called for INSTANCE.square, but very confusing

class TTest4(object):
  def __init__(self):
    self.x= 5

  @property
  def square(self):
    return self.x**2

  @square.setter
  def square(self, value):
    self.x= value**0.5

class TTest5(object):
  def __init__(self):
    self._x= [1,2,3]

  @property
  def x(self):
    return self._x


if __name__=='__main__':
  test1= TTest1()
  print(test1.square)  #result: 4
  #test1.square= 10  #ERROR: 'AttributeError: can't set attribute'
  #print test1.square

  print('-------------')

  test2= TTest2()
  print(test2.square)  #result: 9
  #test2.square= 10  #ERROR: Exception: not writable
  #print test2.square

  print('-------------')

  test3= TTest3()
  print(test3.square)  #result: 64

  print('-------------')

  test4= TTest4()
  print(test4.square)  #result: 25
  test4.square= 10
  print(test4.square)  #result: 10

  print('-------------')

  test5= TTest5()
  print(test5.x)  #result: [1, 2, 3]
  test5.x[1]= 10
  print(test5.x)  #result: [1, 10, 3]

