#!/usr/bin/python3

class TTest:
  x= 10
  def __init__(self):
    self.y= 100
  def __repr__(self):
    return 'x=%r y=%r' % (self.x, self.y)

class TTestC:
  x= 20
  y= TTest()
  def __repr__(self):
    return 'x=%r y={%r}' % (self.x, self.y)

class TTestC2:
  def __init__(self):
    self.x= 20
    self.y= TTest()
  def __repr__(self):
    return 'x=%r y={%r}' % (self.x, self.y)

#test1= TTestC()
#test2= TTestC()
test1= TTestC2()
test2= TTestC2()
test1.x= 209
test1.y.x= 109
test1.y.y= 1009
print('test1=',test1)
print('test2=',test2)

