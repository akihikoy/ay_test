#!/usr/bin/python
#\file    sample_mod.py
#\brief   Example module to test import, reload, etc.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.15, 2017

import datetime

TimeImported= datetime.datetime.now()
print 'Module sample_mod has been imported at', TimeImported

def F():
  print 'F in sample_mod imported at', TimeImported

