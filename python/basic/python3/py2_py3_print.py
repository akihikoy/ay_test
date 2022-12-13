#\file    py2_py3_print.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.13, 2022
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
def Print(*args):
  sys.stdout.write(' '.join(args))
  sys.stdout.write('\n')

if __name__=='__main__':
  try:
    exec('print "Works!","Works!"')
    Print('[print ""] works in the current Python interpreter.')
  except SyntaxError:
    Print('[print ""] does not in the current Python interpreter.')
  try:
    exec('print("Works!","Works!")')
    Print('[print()] works in the current Python interpreter.')
  except SyntaxError:
    Print('[print()] does not in the current Python interpreter.')
