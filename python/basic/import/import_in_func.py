#!/usr/bin/python3
#\file    import_in_func.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jan.29, 2018

def Importer(switch):
  if switch==1:
    from math import *  #SyntaxWarning: import * only allowed at module level
  elif switch==2:
    from time import *  #SyntaxWarning: import * only allowed at module level

if __name__=='__main__':
  try:
    print('pi=',pi)
    print('time=',time())
  except Exception as e:
    print(e)
  print('-----------')
  Importer(1)
  try:
    print('pi=',pi)
    print('time=',time())
  except Exception as e:
    print(e)
  print('-----------')
  Importer(2)
  try:
    print('pi=',pi)
    print('time=',time())
  except Exception as e:
    print(e)
