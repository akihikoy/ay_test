#!/usr/bin/python3
#Class inheritance test

class TBase(object):
  def __init__(self):
    print('super=',super(TBase))
    self.a= 123
    print('Initializing base %r (%r)'%(self.a,self.__class__.__name__))
    self.Print()
    print('----')
  def __del__(self):
    print('Deleting base %r (%r)'%(self.a,self.__class__.__name__))
    self.Print()
    print('----')
  def Print(self):
    print('Base:',self)
    print('   a:',self.a)

class TTest(TBase):
  def __init__(self):
    print('super=',super(TTest))
    super(TTest,self).__init__()
    self.a= 456
  def __del__(self):
    self.a= 789
    super(TTest,self).__del__()
  def Print(self):
    super(TTest,self).Print()
    print('Test:',self)
    print('   a:',self.a)

if __name__=='__main__':
  #base= TBase()
  test= TTest()

  print('=====')
  #base.Print()
  #print '+++++'
  test.Print()
  print('=====')
  #del base
  del test
