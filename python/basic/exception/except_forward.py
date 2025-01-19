#!/usr/bin/python3
import sys
import traceback

def PrintException(e, msg=''):
  print('Exception( %r )%s:' % (type(e), msg))
  print('  ... %r' % (e))
  print('{')
  traceback.print_tb(sys.exc_info()[2])
  print('}')

def BadFunc():
  a=[1,2,3]
  for i in range(4):
    print(a[i])

def SafeExecuter1(func):
  try:
    func()
  except Exception as e:
    PrintException(e, ' caught in SafeExecuter1')
  finally:
    print('#SafeExecuter1 has executed ',func)

def SafeExecuter2(func):
  try:
    func()
  except Exception as e:
    PrintException(e, ' caught in SafeExecuter2')
    raise e  #Forward the exception
  finally:  #NOTE: This part is executed before fowarding (raise e)
    print('#SafeExecuter2 has executed ',func)

if __name__=='__main__':
  try:
    BadFunc()
  except Exception as e:
    PrintException(e, ' caught in __main__-1')
  finally:
    print('#__main__-1 has executed BadFunc()')

  print('---------')

  try:
    SafeExecuter1(BadFunc)
  except Exception as e:
    PrintException(e, ' caught in __main__-2')
  finally:
    print('#__main__-1 has executed SafeExecuter1(BadFunc)')

  print('---------')

  try:
    SafeExecuter2(BadFunc)
  except Exception as e:
    PrintException(e, ' caught in __main__-3')
  finally:
    print('#__main__-1 has executed SafeExecuter2(BadFunc)')
