#!/usr/bin/python3
#\file    capture_print.py
#\brief   Capture the print function and redirect.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.18, 2021
import sys

#This is an example of capturing stdout (print), storing them in a list, and display at the end.
class TStdOutCapturer(object):
  def __enter__(self):
    self._stdout= sys.stdout
    sys.stdout= self
    self.str_list= []
  def __exit__(self, *args):
    sys.stdout= self._stdout
    print(self.str_list)
  def write(self, s):
    if s=='\n':  self.str_list.append('---')
    else:  self.str_list.append(s)
    self._stdout.write(s)
  def flush(self):
    self._stdout.flush()

def Test1():
  print('Test1 is called')
  Test2()
def Test2():
  print('Test2 is called...', end=' ')
  print(1.23, end=' ')
  print('done.')

if __name__=='__main__':
  with TStdOutCapturer():
    print('hello')
    Test1()
