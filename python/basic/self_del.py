#!/usr/bin/python3

class TTest:
  def __init__(self,a):
    self.a= a
    print('initialized',self.a)
  def __del__(self):
    print('deleted',self.a)
  def Destruct(self):
    print('Destruct is called',self.a)
    del self
    self= None

if __name__=='__main__':
  print('------------------')
  test= TTest(999)
  print('------------------')
  print(test)
  print('------------------')
  test.Destruct()  #NOTE: this does not delete the instance
  print('------------------')
  print(test)
  print('------------------')
  del test
  print('------------------')

print('=================')

