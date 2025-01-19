#!/usr/bin/python3
import os
FLAG= True if 'TEST' in os.environ else False
#if not FLAG:  return  #SyntaxError: 'return' outside function
#if not FLAG:
  #import sys
  #sys.exit(0)
if not FLAG:  raise ImportError('TEST is not defined')

def test():
  print('test is defined')
