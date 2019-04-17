#!/usr/bin/python
try:
  import pkg2
except ImportError as e:
  print e

if __name__=='__main__':
  pkg2.test()
