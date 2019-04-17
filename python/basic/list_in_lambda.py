#!/usr/bin/python
import copy

if __name__=='__main__':
  def PrintEq(s):  print '%s= %r' % (s, eval(s))

  a= 2
  f1= lambda:a*a
  PrintEq('a')
  PrintEq('f1()')

  a= 4
  f2= lambda:a**3
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a= 10

  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'

  a= [1,2,3]
  f1= lambda:[e*e for e in a]
  PrintEq('a')
  PrintEq('f1()')

  a= [4,5,6,7,8,9]
  f2= lambda:[e*e*e for e in a]
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a[1]= 10

  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'

  def DefF1():
    a= 2
    return lambda:a*a
  a= 0
  f1= DefF1()
  PrintEq('a')
  PrintEq('f1()')

  def DefF2():
    a= 4
    return lambda:a**3
  f2= DefF2()
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a= 10

  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'

  a= [1,2,3]
  def DefF1(a):
    return lambda:[e*e for e in a]
  f1= DefF1(a)
  a= []
  PrintEq('a')
  PrintEq('f1()')

  a= [4,5,6,7,8,9]
  def DefF2(a):
    return lambda:[e*e*e for e in a]
  f2= DefF2(a)
  a= []
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a= [4,5,6]
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'

  a= 2
  def DefF1(a):
    return lambda:a*a
  f1= DefF1(a)
  a= 0
  PrintEq('a')
  PrintEq('f1()')

  a= 4
  def DefF2(a):
    return lambda:a**3
  f2= DefF2(a)
  a= 0
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a= 10

  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'

  a= 2
  f1= lambda:copy.deepcopy(a)**2
  PrintEq('a')
  PrintEq('f1()')

  a= 4
  f2= lambda:copy.deepcopy(a)**3
  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  a= 10

  PrintEq('a')
  PrintEq('f1()')
  PrintEq('f2()')

  print '---------------'
