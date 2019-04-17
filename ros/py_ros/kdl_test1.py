#!/usr/bin/python
#\file    kdl_test1.py
#\brief   certain python script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.23, 2017
import roslib; roslib.load_manifest('urdfdom_py')
import rospy
import kdl_parser_py.urdf

if __name__=='__main__':
  #cf. https://github.com/ros/kdl_parser/blob/kinetic-devel/kdl_parser_py/kdl_parser_py/urdf.py

  #(ok, tree)= kdl_parser_py.urdf.treeFromFile(filename)
  #(ok, tree)= kdl_parser_py.urdf.treeFromParam('robot_description')

  robot= kdl_parser_py.urdf.urdf.URDF.from_parameter_server('robot_description')
  (ok, tree)= kdl_parser_py.urdf.treeFromUrdfModel(robot)
  assert(ok)

  link_names= [link.name for link in robot.links]
  joint_names= [joint.name for joint in robot.joints]
  joint_parents= [joint.parent for joint in robot.joints]
  joint_limits= [joint.limit for joint in robot.joints if joint.type!='fixed']
  joint_limits_lower= [joint.limit.lower for joint in robot.joints if joint.type!='fixed']
  joint_limits_upper= [joint.limit.upper for joint in robot.joints if joint.type!='fixed']
  joint_limits_vel= [joint.limit.velocity for joint in robot.joints if joint.type!='fixed']
  joint_types= [joint.type for joint in robot.joints]
  num_non_fixed_joints= len(joint_types)-joint_types.count('fixed')

  print 'URDF links:',len(robot.links)
  print 'URDF joints:',len(robot.joints)
  print 'URDF non-fixed joints:',num_non_fixed_joints
  print 'KDL joints:',tree.getNrOfJoints()
  print 'KDL segments:',tree.getNrOfSegments()

  print 'base_link          :',robot.get_root()
  print 'link_names         :',link_names
  print 'joint_names        :',joint_names
  print 'joint_parents      :',joint_parents
  print 'joint_limits       :',joint_limits
  print 'joint_limits_lower :',joint_limits_lower
  print 'joint_limits_upper :',joint_limits_upper
  print 'joint_limits_vel   :',joint_limits_vel
  print 'joint_types        :',joint_types

  import random
  base_link= robot.get_root()
  end_link= link_names[random.randint(0, len(link_names)-1)]
  #end_link= 'link_t'
  chain= tree.getChain(base_link, end_link)
  print "Root link: %s; Random end link: %s" % (base_link, end_link)
  print 'chain.getNrOfSegments():',chain.getNrOfSegments()
  print [chain.getSegment(i).getName() for i in range(chain.getNrOfSegments())]
  print [chain.getSegment(i).getJoint().getName() for i in range(chain.getNrOfSegments())]

