#!/usr/bin/python
import os
import readline

T=0
#def Init(t):
  #global T
  #T= t

class Test:
  def __init__(self,a):
    self.a= a
  def Func1(self):
    print '###',self.a
  def Func2(self):
    print '@@@',self.a
  def Func3(self):
    print 'XXX',self.a

  def Load(self,i):
    a='sub.sub'+str(i)
    sub= __import__(a,globals(),locals(),a,-1)
    print sub
    #sub.Init(self)
    sub.Run(self)

if __name__=='__main__':
  t= Test('hoge')
  #Init(t)
  T= t
  print t
  print T
  t.Func1()
  T.Func1()
  t.a= 'hehe'
  t.Func1()
  T.Func1()
  #i= 1
  #a='sub.sub'+str(i)
  #__import__(a)
  t.Load(1)
  t.a= 'hyaaaaa'
  t.Load(1)
