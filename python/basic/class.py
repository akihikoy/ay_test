#!/usr/bin/python

class TTest:
  x= 10
  def __init__(self):
    self.y= None
  def __repr__(self):
    return 'x=%r y=%r' % (self.x, self.y)

class TTestB:
  x= 10
  y= None

test1= TTest()
test1.x= 20
test1.y= 200
test2= TTest()
print 'test1=',test1
print 'test2=',test2
TTest.x= 30
TTest.y= 300
test3= TTest()
print 'test3=',test3

print '-----'

test1= TTestB()
test1.x= 20
test1.y= 200
test2= TTestB()
print 'test1=',test1.x,test1.y
print 'test2=',test2.x,test2.y
TTestB.x= 30
TTestB.y= 300
test3= TTestB()
print 'test2=',test3.x,test3.y
