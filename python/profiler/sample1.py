#!/usr/bin/python
#\file    sample1.py
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\date    Jun.04, 2015

def Func1():
  d= 0
  for i in range(1000000):
    d= d+i*i*i*i
  print d

def Func2():
  d= 0
  for i in range(1000000):
    d= d+i**4
  print d

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))
  Func1()
  Func2()
