#!/usr/bin/python
#\file    __init__.py
#\brief   Test of python packages
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Aug.29, 2015

import sys
using_ros= ('ros' in ' '.join(sys.path))

import os
robot= os.environ['ROBOT'] if 'ROBOT' in os.environ else None

print 'using_ros=',using_ros
print 'robot=',robot

from pkg1 import *
if robot=='baxter':
  from pkg2 import *

