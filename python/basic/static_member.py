#!/usr/bin/python

def Init(a):
  print 'Initializing with',a
  return a

class TTest:
  A= Init(10)


if __name__=='__main__':
  print '-------'
  test= TTest()
  print test.A, TTest.A

  test.A= 20
  print test.A, TTest.A

  TTest.A= 30
  print test.A, TTest.A

