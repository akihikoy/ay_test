#!/usr/bin/python
#\file    locals1.py
#\brief   Check strange behavior of locals.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.19, 2017

'''
http://stackoverflow.com/questions/44075902/python-strange-behavior-of-locals
NOTE: locals() doesn't just return a dict of local variables; it also updates the dict to reflect current local variable values.
NOTE: Local variables aren't actually implemented with a dict, so no.
'''

def Main1():
  print id(locals()), locals()
  def F(l=locals()):  print id(l), l
  a= 100
  #locals()['b']= 100
  F()
  print id(locals()), locals()

def Main2():
  print id(locals()), locals()
  def F(l=locals()):  print id(l), l
  a= 100
  locals()['b']= 100
  F()
  print id(locals()), locals()

def Main3():
  def F(l=locals()):  print 'F', id(l), l
  a= 100
  F()
  print '1', id(locals()), locals()
  F()

if __name__=='__main__':
  Main1()
  print '-----'
  Main2()
  print '-----'
  Main3()
  print '-----'
  def F(g=globals()):  print 'F', id(g), g
  a= 100
  F()
  print '1', id(globals()), globals()
  F()

