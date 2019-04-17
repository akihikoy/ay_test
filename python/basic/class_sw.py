#!/usr/bin/python
#\file    class_sw.py
#\brief   Switching two classes.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.01, 2017

class TTest1(object):
  def __init__(self):
    self.A= 1

class TTest2(object):
  def __init__(self):
    self.A= 10

'''
class TTest(object):
  def __init__(self,type):
    return TTest1() if type==1 else TTest2()  #ERROR: This is impossible.
'''

def TTest(type):
  return TTest1() if type==1 else TTest2()  #ERROR: This is impossible.

if __name__=='__main__':
  test1= TTest(1)
  print 'test1=',test1
  test2= TTest(2)
  print 'test2=',test2
