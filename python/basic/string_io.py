#!/usr/bin/python3
#\file    string_io.py
#\brief   Capture stdout.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.18, 2021
#from io import StringIO
#from io import BytesIO as StringIO  #For Py2
from io import StringIO  #For Py3
import sys

#Capture print (stdout) as a list.
#src: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list):
  def __enter__(self):
    self._stdout= sys.stdout
    sys.stdout= self._stringio= StringIO()
    return self
  def __exit__(self, *args):
    self.extend(self._stringio.getvalue().splitlines())
    del self._stringio    # free up some memory
    sys.stdout= self._stdout

if __name__=='__main__':
  #Capture print (stdout) as a list:
  with Capturing() as output:
    print('hello world')
  print('captured:',output)


