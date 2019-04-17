#!/usr/bin/python
import gc
class TTest:
  def __init__(self):
    self.sub_func= None
    print 'Created',self
  def __del__(self):
    self.sub_func= None
    print 'Deleted',self
  def Print(self):
    print 'Print',self

def SubFunc1(t):
  t.Print()
def DefineObj1():
  t= TTest()
  t.sub_func= lambda: SubFunc1(t)
  return t

t= DefineObj1()
t.sub_func()
#t.sub_func= None
#gc.collect()
#print gc.garbage
del t
#gc.collect()
#print gc.garbage

print '--------'

def DefineObj2():
  t= TTest()
  def SubFunc2():
    t.Print()
  t.sub_func= SubFunc2
  return t

t= DefineObj2()
t.sub_func()
#t.sub_func= None
del t

print '--------'

import weakref

def SubFunc1_2(t):  #== SubFunc1
  t.Print()
def DefineObj1_2():
  t= TTest()
  tr= weakref.ref(t)
  t.sub_func= lambda: SubFunc1_2(tr())
  return t

t= DefineObj1_2()
t.sub_func()
del t

print '--------'

def DefineObj2_2():
  t= TTest()
  tr= weakref.ref(t)
  def SubFunc2_2():  #!=SubFunc2
    tr().Print()
  t.sub_func= SubFunc2_2
  return t

t= DefineObj2_2()
t.sub_func()
#t.sub_func= None
del t

print '--------'

def DefineObj2_3():
  t= TTest()
  tr= weakref.ref(t)
  def SubFunc2_3(t):  #!=SubFunc2
    t.Print()
  t.sub_func= lambda: SubFunc2_3(tr())
  return t

t= DefineObj2_3()
t.sub_func()
#t.sub_func= None
del t

print '--------'

def ForceDelete(obj, exclude=[]):
  for (k,v) in obj.__dict__.iteritems():
    if not k in exclude:
      obj.__dict__[k]= None

t= DefineObj1()
t.sub_func()
ForceDelete(t)
del t

print '--------'

import types

  #def RegisterFunc(self,func,name):
    #self.__dict__[name]= types.MethodType(func,self)

#def SubFunc1_3(self):
  #self.Print()
#def DefineObj1_3():
  #t= TTest()
  #t.RegisterFunc(SubFunc1_3,'sub_func')
  ##t.sub_func= SubFunc1_3
  #return t

#t= DefineObj1_3()
#t.sub_func()
#del t

#print '--------'
