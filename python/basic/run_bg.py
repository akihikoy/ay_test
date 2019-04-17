#!/usr/bin/python
#\file    run_bg.py
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\date    Jun.03, 2015
import subprocess

if __name__=='__main__':
  #def PrintEq(s):  print '%s= %r' % (s, eval(s))

  stderr_redirect=subprocess.PIPE
  p= subprocess.Popen('kwrite',shell=True,stdin=subprocess.PIPE,stderr=stderr_redirect)
  print 'kwrite has been launched'

