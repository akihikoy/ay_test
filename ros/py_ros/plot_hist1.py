#!/usr/bin/python
#\file    plot_hist1.py
#\brief   Plot histogram of a list of topics.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Apr.22, 2024
'''
NOTE:
Implemented as a debugger tool for ROS topics.
e.g.
$ ./plot_hist1.py /grasp_plan grasp_list FingerCollisionScore
'''
import roslib
import rospy
import rostopic
import rospkg
import importlib
import os, sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

#Values to plot.
Values= [None]

#def PrintMsgContent(msg, indent=0):
  #for k in dir(msg):
    #if not k.startswith('_'):
      #v= getattr(msg, k)
      #if inspect.isclass(v):
        #print '{}{}<{}>:'.format(' '*indent,k,type(v))
        #PrintMsgContent(v, indent+2)
      #else:
        #print '{}{}: {}'.format(' '*indent,k,type(v))

def CallbackTopic(msg, list_names, value_names):
  try:
    list_value= msg
    for name in list_names:
      list_value= getattr(list_value, name)
  except AttributeError:
    print 'Failed to find list_names={}'.format(list_names)
    #print 'Message content:',dir(msg),dir(msg.__class__)
    rospy.signal_shutdown('Exception')
  assert(isinstance(list_value,list))
  def get_value(elem):
    for name in value_names:
      elem= getattr(elem,name)
    return elem
  try:
    #print list_value[0] if len(list_value)>0 else None
    values= [get_value(elem) for elem in list_value]
    #print values
  except AttributeError:
    print 'Failed to find value_names={}'.format(value_names)
    rospy.signal_shutdown('Exception')
  Values[0]= values

if __name__=='__main__':
  rospy.init_node('plot_hist1')
  topic_name= sys.argv[1]
  list_names= sys.argv[2].split('/')
  value_names= sys.argv[3].split('/')

  # Determine the topic type
  topic_type= rostopic.get_topic_type(topic_name)[0]
  if topic_type is None:
      raise Exception('Topic {} does not exist or is not active.'.format(topic_name))

  # Dynamically import the module and class based on topic type
  module_name,class_name= topic_type.split('/')

  print 'topic_name= {}'.format(topic_name)
  print 'module_name= {}'.format(module_name)
  print 'class_name= {}'.format(class_name)
  print 'list_names= {}'.format(list_names)
  print 'value_names= {}'.format(value_names)

  rospack= rospkg.RosPack()
  # Get the path for the specified package and add it to sys.path if not already included
  try:
    pkg_path= os.path.join(rospack.get_path(module_name), 'src')
    print 'pkg_path= {}'.format(pkg_path)
    if pkg_path not in sys.path:
      sys.path.append(pkg_path)
  except rospkg.common.ResourceNotFound:
    raise ImportError('Package {} not found in the ROS environment.'.format(module_name))

  ros_pkg= importlib.import_module(module_name+'.msg')
  msg_class= getattr(ros_pkg, class_name)

  sub= rospy.Subscriber(topic_name, msg_class, lambda msg:CallbackTopic(msg,list_names,value_names))

  plt.rcParams['keymap.quit'].append('q')

  rate_adjuster= rospy.Rate(30)
  xmin,xmax,ymax= None,None,None
  while not rospy.is_shutdown():
    plt.cla()
    if Values[0] is not None and len(Values[0])>0:
      values= Values[0]
      xmin= min(xmin,np.min(values)) if xmin is not None else np.min(values)
      xmax= max(xmax,np.max(values)) if xmax is not None else np.max(values)
      n,bins,patches= plt.hist(values, bins=50, density=False, facecolor='g', alpha=0.75, label='hist')
      ymax= max(ymax,np.max(n)) if ymax is not None else np.max(n)
      plt.title('Histogram')
      plt.xlabel('/'.join(value_names))
      plt.ylabel('Frequency')
      plt.xlim(left=xmin, right=xmax)
      plt.ylim(bottom=None, top=ymax)
      plt.legend()
    rate_adjuster.sleep()
    plt.pause(0.01)

  if Values[0] is None:
    print '============Print debug info============'
    cmd= 'rostopic info {}'.format(topic_name)
    print 'topic info ({}):'.format(cmd)
    os.system(cmd)
    cmd= 'rosmsg info {}/{}'.format(module_name,class_name)
    print 'message info ({}):'.format(cmd)
    os.system(cmd)

  sub.unregister()
  rospy.signal_shutdown('Quit')


