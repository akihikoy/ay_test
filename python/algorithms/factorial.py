#!/usr/bin/python

if __name__=='__main__':
  factorial= lambda n:1 if n==1 else n*factorial(n-1)

  print '3!=',factorial(3)
  print '9!=',factorial(9)
  print '15!=',factorial(15)

