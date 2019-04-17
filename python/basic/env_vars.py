#!/usr/bin/python
#\file    env_vars.py
#\brief   Environment variables.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.29, 2017

import os
print '$HOME=',os.environ['HOME']
print '$ROS*='
for key,value in {key:value for key,value in os.environ.iteritems() if key[:3]=='ROS'}.iteritems():
  print '  ${key}= {value}'.format(key=key,value=value)

