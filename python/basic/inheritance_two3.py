#!/usr/bin/python
#\file    inheritance_two3.py
#\brief   Class inheritance test of two super classes.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.05, 2022

class TBaseA(object):
  def __init__(self, a=100):
    print 'TBaseA.__init__ is called with a={}'.format(a)
    self.a=a

class TBaseB(object):
  def __init__(self, a=200):
    print 'TBaseB.__init__ is called with a={}'.format(a)
    self.a=a*10

class TTest1(TBaseA,TBaseB):
  def __init__(self):
    print 'TTest1.__init__ checkpoint #1 / a=N/A'
    super(TTest1,self).__init__()
    print 'TTest1.__init__ checkpoint #2 / a={}'.format(self.a)
    super(TTest1,self).__init__(a=300)
    print 'TTest1.__init__ checkpoint #3 / a={}'.format(self.a)
    TBaseA.__init__(self,a=400)
    print 'TTest1.__init__ checkpoint #4 / a={}'.format(self.a)
    TBaseB.__init__(self,a=500)
    print 'TTest1.__init__ checkpoint #5 / a={}'.format(self.a)

if __name__=='__main__':
  print 'start'
  test= TTest1()
  print 'done'
