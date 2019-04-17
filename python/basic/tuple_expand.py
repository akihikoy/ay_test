#!/usr/bin/python

def Test1(a,b,c):
  print '------------'
  print 'a:',a
  print 'b:',b
  print 'c:',c

def Test2(*args):
  print '------------'
  print 'len(args):',len(args)
  print 'args:',args

abc=(0.1, 2, 3)
#Test1(abc)  #ERROR
Test1(*abc)
Test2(abc)  #len(args):1
Test2(*abc)  #len(args):3
