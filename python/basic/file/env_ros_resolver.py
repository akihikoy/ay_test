#!/usr/bin/python3
#\file    env_ros_resolver.py
#\brief   Resolving function test of environmental variables and ROS package path.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Jun.05, 2025
import re, os
import rospkg

'''
Convert environmental variable and ROS package reference in str s.
s can include ${env XXX} and ${pkg YYY} to refer an environmental variable XXX
and a ROS package YYY.
'''
def ResolveEnvROSPkg(s):
  f= ResolveEnvROSPkg
  if not hasattr(f, 'env_var_pattern'):
    f.env_var_pattern= re.compile(r'\$\{env\s+([A-Za-z0-9_]+)\}')
  if not hasattr(f, 'ros_pkg_pattern'):
    f.ros_pkg_pattern= re.compile(r'\$\{pkg\s+([A-Za-z0-9_]+)\}')
  s= f.env_var_pattern.sub(lambda m: os.getenv(m.group(1), m.group(0)), s)
  s= f.ros_pkg_pattern.sub(lambda m: rospkg.RosPack().get_path(m.group(1)), s)
  return s

if __name__=='__main__':
  dir1= '${pkg maskrcnn_streaming2}/maskrcnn_streaming/ros_script/output/'
  dir2= '${env HOME}/src/${env USER}/'

  print(f'dir1 = {dir1} = {ResolveEnvROSPkg(dir1)}')
  print(f'dir2 = {dir2} = {ResolveEnvROSPkg(dir2)}')
