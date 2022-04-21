#!/usr/bin/python
#\file    time_str2.py
#\brief   Time to string.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.21, 2022
import datetime

def TimeStr(fmt='short2', now=None):
  if now is None:  now= datetime.datetime.utcnow()
  if fmt=='normal':  return now.strftime('%Y.%m.%d-%H.%M.%S')
  if fmt=='normal_ms':  return now.strftime('%Y.%m.%d-%H.%M.%S.%f')[:-3]
  if fmt=='normal_us':  return now.strftime('%Y.%m.%d-%H.%M.%S.%f')
  if fmt=='short':   return now.strftime('%Y%m%d%H%M%S')
  if fmt=='short2':  return now.strftime('%Y%m%d-%H%M%S')
  if fmt=='short3':  return now.strftime('%Y%m%d-%H%M%S-%f')
  raise Exception('TimeStr: Invalid format:', fmt)

if __name__=='__main__':
  import time
  now= datetime.datetime.utcnow()
  print 'datetime.datetime.utcnow:', now
  print 'time.time()*1e6:', time.time()*1e6
  print 'short:', TimeStr('short',now)
  print 'short2:', TimeStr('short2',now)
  print 'short3:', TimeStr('short3',now)
  print 'normal:', TimeStr('normal',now)
  print 'normal_ms:', TimeStr('normal_ms',now)
  print 'normal_us:', TimeStr('normal_us',now)
