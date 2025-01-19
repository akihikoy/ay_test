#!/usr/bin/python3
#\file    finally1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jul.12, 2018

def func():
  print('started')
  try:
    print('trying')
    print(0/0)
  except Exception as e:
    print('Exception:',e)
    raise e  #Forwarding exception.
  finally:
    print('Finally code')
  print('Code after Finally')

func()
print('Main code after func')
