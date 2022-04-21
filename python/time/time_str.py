#!/usr/bin/python
#\file    time_str.py
#\brief   Time to string.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Sep.06, 2021
import time

#def TimeStr(fmt='short2', now=None):
  #if now is None:  now= time.localtime()
  #if fmt=='normal':  return '%04i.%02i.%02i-%02i.%02i.%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  #if fmt=='short':   return '%04i%02i%02i%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  #if fmt=='short2':  return '%04i%02i%02i-%02i%02i%02i' % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  #raise Exception('TimeStr: Invalid format:', fmt)

def TimeStr(fmt='short2', now=None):
  if now is None:  now= time.localtime()
  if fmt=='normal':  return '{:04d}.{:02d}.{:02d}-{:02d}.{:02d}.{:02d}'.format(now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  if fmt=='short':   return '{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  if fmt=='short2':  return '{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}'.format(now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)
  raise Exception('TimeStr: Invalid format:', fmt)

if __name__=='__main__':
  now= time.localtime()
  print 'time.localtime:', now
  print 'time.time()*1e6:', time.time()*1e6
  print 'short:', TimeStr('short',now)
  print 'short2:', TimeStr('short2',now)
  print 'normal:', TimeStr('normal',now)
