#!/usr/bin/python3
#\file    time_str_py3.py
#\brief   Time to string.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.21, 2022
import time

def TimeStr(fmt='short2', now=None):
  if now is None:  now= time.localtime()
  if fmt=='normal':  return f'{now.tm_year:04d}.{now.tm_mon:02d}.{now.tm_mday:02d}-{now.tm_hour:02d}.{now.tm_min:02d}.{now.tm_sec:02d}'
  if fmt=='short':   return f'{now.tm_year:04d}{now.tm_mon:02d}{now.tm_mday:02d}{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}'
  if fmt=='short2':  return f'{now.tm_year:04d}{now.tm_mon:02d}{now.tm_mday:02d}-{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}'
  raise Exception('TimeStr: Invalid format:', fmt)

if __name__=='__main__':
  now= time.localtime()
  print('time.localtime:', now)
  print('time.time()*1e6:', time.time()*1e6)
  print('short:', TimeStr('short',now))
  print('short2:', TimeStr('short2',now))
  print('normal:', TimeStr('normal',now))
