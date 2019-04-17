#!/usr/bin/python

##x does not change:
#def Test1(x):
  #x*= 10.0
#def Test2(x):
  #x= 20.0
#class TTest:
  #def __init__(self):
    #self.x= 0.0
  #def Test1(self):
    #self.x*= 10.0
  #def Test2(self):
    #self.x= 20.0

#x changes:
def Test1(x):
  x[0]*= 10.0
def Test2(x):
  x[0]= 20.0
class TTest:
  def __init__(self):
    self.x= 0.0
  def Test1(self):
    self.x[0]*= 10.0
  def Test2(self):
    self.x[0]= 20.0

#class Ref()

def PrintEq(s):  print '%s= %r' % (s, eval(s))
if __name__=='__main__':
  #x= 1
  x= [1]
  PrintEq('x')
  Test1(x)
  PrintEq('x')
  Test2(x)
  PrintEq('x')
  test= TTest()
  test.x= x
  test.Test1()
  PrintEq('x')
  test.Test2()
  PrintEq('x')
