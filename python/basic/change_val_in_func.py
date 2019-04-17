#!/usr/bin/python

class TWrapper:
  def __init__(self,value):
    self.Value= value
  def __repr__(self):
    return str(self.Value)
  def __str__(self):
    return str(self.Value)
  def __getitem__(self,key):
    return self.Value
  def __setitem__(self,key,value):
    self.Value= value

def func1(x):
  x= 100

def func2(x):
  x[0]= 100
  #x.append(99)

def func2b(x):
  x[None]= 100

def func3(x):
  x= []
  #x.append(99)

def func4(x):
  x[:]= []
  #x.append(99)

a=1
b=2
b2=2
b3=2
b4=TWrapper(2)
c=[3]
d=[1,2,3]
e=[1,2,3]
print 'a=',a
print 'b=',b
print 'b2=',b2
print 'b3=',b3
print 'b4=',b4
print 'c=',c
print 'd=',d
print 'e=',e

func1(a)
func2([b])
b2w=[b2]; func2(b2w)
func2b(TWrapper(b3))
func2b(b4)
func2(c)
func3(d)
func4(e)

print 'a=',a
print 'b=',b
print 'b2=',b2
print 'b3=',b3
print 'b4=',b4
print 'c=',c
print 'd=',d
print 'e=',e
