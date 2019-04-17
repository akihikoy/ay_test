#!/usr/bin/python
#\file    import_lfd_trick.py
#\brief   TEST of importing package from lfd_trick on non-ROS program.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2017

import sys,os
sys.path.append(os.environ['HOME']+'/ros_ws/lfd_trick/src')

import cma
print 'cma.CMAEvolutionStrategy=', cma.CMAEvolutionStrategy

from base.base_util import *
print 'AskGen Test:'
print AskGen('a','b','c')

#print dir()

from base.base_geom import *
print '12.5 [rad]=', AngleMod1(12.5)
