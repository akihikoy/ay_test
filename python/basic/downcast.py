#!/usr/bin/python
#\file    downcast.py
#\brief   Can we do downcast in python?
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.09, 2015

#cf. http://stackoverflow.com/questions/15187653/how-do-i-downcast-in-python

class TA(object):
  def __init__(self):
    self.X= 'aaa'

class TB(TA):
  def __init__(self):
    TA.__init__(self)
  def Do(self):
    print 'X is', self.X

a= TA()
a.__class__= TB  #DOWNCAST
a.Do()
