#!/usr/bin/python
# ref. http://docs.python.jp/2/library/sys.html#sys.exc_traceback
# ref. http://docs.python.jp/2/library/sys.html#sys.exc_info
# http://docs.python.jp/2/library/traceback.html
import sys
import traceback

try:
  a=[1,2,3]
  for i in range(4):
    print a[i]
except Exception as e:
  print 'Error'
  print '  e: ',e
  print '  type: ',type(e)
  print '  args: ',e.args
  print '  message: ',e.message
  print '  sys.exc_info(): ',sys.exc_info()

  #print '  sys.exc_traceback.tb_lineno: ',sys.exc_traceback.tb_lineno
  #print '  sys.exc_traceback: ',sys.exc_traceback
  #WARNING: do not use exc_traceback, which is duplicated; use exc_info instead

  print '  sys.exc_info()[1]: ',sys.exc_info()[1]
  print '  sys.exc_info()[2]: ',sys.exc_info()[2]
  print '  sys.exc_info()[2].tb_lineno: ',sys.exc_info()[2].tb_lineno
  print '  traceback.print_tb(sys.exc_info()[2]): '
  traceback.print_tb(sys.exc_info()[2])

  print '-----------'
  #Nice format:
  print 'Error(',type(e),'):'
  print '  ',e
  #print '  type: ',type(e)
  #print '  args: ',e.args
  #print '  message: ',e.message
  #print '  sys.exc_info(): ',sys.exc_info()
  print '  Traceback: '
  traceback.print_tb(sys.exc_info()[2])

print '-----------'

a=[1,2,3]
for i in range(4):
  print a[i]

