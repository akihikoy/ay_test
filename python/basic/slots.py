#!/usr/bin/python3
#\file    slots.py
#\brief   Test of slots
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.23, 2015

#refs:
  #http://docs.python.jp/2/reference/datamodel.html#slots
  #http://www.saltycrane.com/blog/2012/08/python-data-object-motivated-desire-mutable-namedtuple-default-values/


class TTest:
  __slots__=('aaa','bbb','ccc')
  def __init__(self):
    self.aaa= 1.2
    self.bbb= [1,[2,3]]
    self.ccc= 'hoge hoge'
    #self.ddd= [3,3,3]  #allowed with Py2, inhibited with Py3.
  def vars(self):
    return {k:getattr(self,k) for k in self.__slots__}

def Main():
  test1= TTest()
  test1.aaa= 3.14
  test1.bbb= [1]
  #test1.ddd= [9,9,9,9]
  #print('test1=',test1.__dict__)  #allowed with Py2, inhibited with Py3.
  print('test1=',test1.vars())
  test2= test1
  test2.ccc= 'hehehe'
  #test2.ddd[0]= 10
  #print('test2=',test2.__dict__)
  print('test2=',test2.vars())
  #print('test1=',test1.__dict__)
  print('test1=',test1.vars())


if __name__=='__main__':
  import sys
  if len(sys.argv)>1 and sys.argv[1] in ('p','plot','Plot','PLOT'):
    PlotGraphs()
    sys.exit(0)
  Main()
