#!/usr/bin/python
#\file    with.py
#\brief   Test with
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Nov.30, 2016

class TTest:
  def __init__(self, n):
    self.N= n
  def __enter__(self, *args, **kwargs):
    print 'Entered',self.N,args,kwargs
    return self
    #NOTE: If __enter__ returns self and it is assigned into a variable at with XXX as YY,
    # this object (==YY) will not be deleted at the end of the with block.
    # Otherwise (__enter__ does not return self or it is not assigned at with XXX),
    # this object will be deleted at the end of the with block.
  def __exit__(self, *args, **kwargs):
    print 'Exited',self.N,args,kwargs
  def __del__(self):
    print 'Deleted',self.N

if __name__=='__main__':
  print 'p1'
  test= TTest('A')
  print 'p2'
  with test:
    print 'hoge 1',test.N

  print 'p3'
  with TTest('B'):
    print 'hoge 2'

  print 'p4'
  with TTest('C') as t3:
    print 'hoge 3'

  print 'p5'
