#!/usr/bin/python
#\file    rviz1.py
#\brief   RVIZ tool, copied from lfd_trick/src/base/ros_viz.py
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.09, 2015

import roslib; roslib.load_manifest('rospy')
import rospy
import tf
import visualization_msgs.msg
import geometry_msgs.msg
import numpy as np
import numpy.linalg as la

#Return a normalized vector with L2 norm
def Normalize(x):
  return np.array(x)/la.norm(x)

#Convert x to geometry_msgs/Pose
def XToGPose(x):
  pose= geometry_msgs.msg.Pose()
  pose.position.x= x[0]
  pose.position.y= x[1]
  pose.position.z= x[2]
  pose.orientation.x= x[3]
  pose.orientation.y= x[4]
  pose.orientation.z= x[5]
  pose.orientation.w= x[6]
  return pose

#3x3 rotation matrix to quaternion
def RotToQ(R):
  M = tf.transformations.identity_matrix()
  M[:3,:3] = R
  return tf.transformations.quaternion_from_matrix(M)

#Convert a pose, x,y,z,quaternion(qx,qy,qz,qw) to pos (x,y,z) and 3x3 rotation matrix
def XToPosRot(x):
  p = np.array(x[0:3])
  R = tf.transformations.quaternion_matrix(x[3:7])[:3,:3]
  return p, R

#Convert pos p=(x,y,z) and 3x3 rotation matrix R to a pose, x,y,z,quaternion(qx,qy,qz,qw)
def PosRotToX(p,R):
  M = tf.transformations.identity_matrix()
  M[:3,:3] = R
  x = list(p)+[0.0]*4
  x[3:7] = tf.transformations.quaternion_from_matrix(M)
  return x

#Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
#original_norm: keep original vec's norm, otherwise the norm is 1.
#Using The Gram-Schmidt process: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def Orthogonalize(vec, base, original_norm=True):
  base= np.array(base)/la.norm(base)
  vec2= vec - np.dot(vec,base)*base
  if original_norm:  return vec2 / la.norm(vec2) * la.norm(vec)
  else:              return vec2 / la.norm(vec2)

#Get an orthogonal axis of a given axis
#preferable: preferable axis (orthogonal axis is close to this)
#fault: return this axis when dot(axis,preferable)==1
def GetOrthogonalAxisOf(axis,preferable=[0.0,0.0,1.0],fault=None):
  axis= np.array(axis)/la.norm(axis)
  if fault is None or 1.0-abs(np.dot(axis,preferable))>=1.0e-6:
    return Orthogonalize(preferable,base=axis,original_norm=False)
  else:
    return fault


class TSimpleVisualizer:
  def __init__(self, viz_dt=rospy.Duration(), name_space='visualizer'):
    self.viz_pub= rospy.Publisher('visualization_marker', visualization_msgs.msg.Marker)
    self.curr_id= 0
    self.added_ids= set()
    self.viz_frame= 'torso_lift_link'
    self.viz_ns= name_space
    self.viz_dt= viz_dt
    #self.viz_dt= rospy.Duration()
    #ICol:r,g,b,  y,p,sb, w
    self.indexed_colors= [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

  def __del__(self):
    if self.viz_dt in (None, rospy.Duration()):
      self.DeleteAllMarkers()
    self.Reset()
    self.viz_pub.publish()
    self.viz_pub.unregister()

  def Reset(self, viz_dt=None):
    self.curr_id= 0
    if viz_dt!=None:
      self.viz_dt= viz_dt

  def DeleteAllMarkers(self):
    #print '[Viz]Deleting all markers:',self.added_ids
    marker= visualization_msgs.msg.Marker()
    for mid in self.added_ids:
      marker.header.frame_id= self.viz_frame
      marker.ns= self.viz_ns
      marker.id= mid
      marker.action= visualization_msgs.msg.Marker.DELETE
      self.viz_pub.publish(marker)
    self.added_ids= set()

  def ICol(self, i):
    return self.indexed_colors[i%len(self.indexed_colors)]

  def GenMarker(self, x, scale, rgb, alpha):
    marker= visualization_msgs.msg.Marker()
    marker.header.frame_id= self.viz_frame
    marker.header.stamp= rospy.Time.now()
    marker.ns= self.viz_ns
    #marker.id= self.curr_id
    marker.action= visualization_msgs.msg.Marker.ADD  # or DELETE
    marker.lifetime= self.viz_dt
    marker.scale.x= scale[0]
    marker.scale.y= scale[1]
    marker.scale.z= scale[2]
    marker.color.a= alpha
    marker.color.r = rgb[0]
    marker.color.g = rgb[1]
    marker.color.b = rgb[2]
    marker.pose= XToGPose(x)
    #self.curr_id+= 1
    return marker

  def SetID(self, marker, mid):
    if mid==None:
      marker.id= self.curr_id
      self.curr_id+= 1
    else:
      marker.id= mid
      if marker.id>=self.curr_id:
        self.curr_id= marker.id+1
    self.added_ids= self.added_ids.union([marker.id])
    return marker.id+1

  #Visualize a marker at x.  If mid==None, the id is automatically assigned
  def AddMarker(self, x, scale=[0.02,0.02,0.004], rgb=[1,1,1], alpha=1.0, mid=None):
    marker= self.GenMarker(x, scale, rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.CUBE  # or CUBE, SPHERE, ARROW, CYLINDER
    self.viz_pub.publish(marker)
    return mid2

  #Visualize an arrow at x.  If mid==None, the id is automatically assigned
  def AddArrow(self, x, scale=[0.05,0.002,0.002], rgb=[1,1,1], alpha=1.0, mid=None):
    marker= self.GenMarker(x, scale, rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.ARROW  # or CUBE, SPHERE, ARROW, CYLINDER
    self.viz_pub.publish(marker)
    return mid2

  #Visualize a cube at x.  If mid==None, the id is automatically assigned
  def AddCube(self, x, scale=[0.05,0.03,0.03], rgb=[1,1,1], alpha=1.0, mid=None):
    marker= self.GenMarker(x, scale, rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.CUBE  # or CUBE, SPHERE, ARROW, CYLINDER
    self.viz_pub.publish(marker)
    return mid2

  #Visualize a sphere at p=[x,y,z].  If mid==None, the id is automatically assigned
  def AddSphere(self, p, scale=[0.05,0.05,0.05], rgb=[1,1,1], alpha=1.0, mid=None):
    if len(p)==3:
      x= list(p)+[0,0,0,1]
    else:
      x= p
    marker= self.GenMarker(x, scale, rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.SPHERE  # or CUBE, SPHERE, ARROW, CYLINDER
    self.viz_pub.publish(marker)
    return mid2

  #Visualize a cylinder whose end points are p1 and p2.  If mid==None, the id is automatically assigned
  def AddCylinder(self, p1, p2, diameter, rgb=[1,1,1], alpha=1.0, mid=None):
    ez= Normalize(Vec(p2)-Vec(p1))
    ex= GetOrthogonalAxisOf(ez,preferable=[1.0,0.0,0.0],fault=[0.0,1.0,0.0])
    ey= np.cross(ez,ex)
    x= [0]*7
    x[0:3]= 0.5*(Vec(p1)+Vec(p2))
    x[3:]= RotToQ(np.matrix([ex,ey,ez]).T)
    length= la.norm(Vec(p2)-Vec(p1))

    scale= [diameter,diameter,length]
    marker= self.GenMarker(x, scale, rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.CYLINDER  # or CUBE, SPHERE, ARROW, CYLINDER
    self.viz_pub.publish(marker)
    return mid2

  #Visualize a points [[x,y,z]*N].  If mid==None, the id is automatically assigned
  def AddPoints(self, points, scale=[0.03,0.03], rgb=[1,1,1], alpha=1.0, mid=None):
    x= [0,0,0, 0,0,0,1]
    marker= self.GenMarker(x, list(scale)+[0.0], rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.POINTS
    for p in points:
      gp= geometry_msgs.msg.Point()
      gp.x= p[0]
      gp.y= p[1]
      gp.z= p[2]
      marker.points.append(gp)
    self.viz_pub.publish(marker)
    return mid2

  #Visualize a coordinate system at x.  If mid==None, the id is automatically assigned
  def AddCoord(self, x, scale=[0.05,0.002], alpha=1.0, mid=None):
    scale= [scale[0],scale[1],scale[1]]
    p,R= XToPosRot(x)
    Ry= np.array([R[:,1],R[:,2],R[:,0]]).T
    Rz= np.array([R[:,2],R[:,0],R[:,1]]).T
    mid= self.AddArrow(x, scale=scale, rgb=self.ICol(0), alpha=alpha, mid=mid)
    mid= self.AddArrow(PosRotToX(p,Ry), scale=scale, rgb=self.ICol(1), alpha=alpha, mid=mid)
    mid= self.AddArrow(PosRotToX(p,Rz), scale=scale, rgb=self.ICol(2), alpha=alpha, mid=mid)
    return mid

  #Visualize a polygon [[x,y,z]*N].  If mid==None, the id is automatically assigned
  def AddPolygon(self, points, scale=[0.02], rgb=[1,1,1], alpha=1.0, mid=None):
    x= [0,0,0, 0,0,0,1]
    marker= self.GenMarker(x, list(scale)+[0.0,0.0], rgb, alpha)
    mid2= self.SetID(marker,mid)
    marker.type= visualization_msgs.msg.Marker.LINE_STRIP
    for p in points:
      gp= geometry_msgs.msg.Point()
      gp.x= p[0]
      gp.y= p[1]
      gp.z= p[2]
      marker.points.append(gp)
    self.viz_pub.publish(marker)
    return mid2

  #Visualize contacts which should be an arm_navigation_msgs/ContactInformation[]
  def AddContacts(self, contacts, scale=[0.01], rgb=[1,1,0], alpha=0.7, mid=None):
    if len(contacts)==0:  return self.curr_id
    for c in contacts:
      p= [c.position.x, c.position.y, c.position.z]
      self.viz_frame= c.header.frame_id
      mid= self.AddSphere(p+[0,0,0,1], scale=scale*3, rgb=rgb, alpha=0.7, mid=mid)
    return mid

#Visualize contacts which should be a arm_navigation_msgs/ContactInformation[]
#NOTE: not efficient, for debug
def VisualizeContacts(contacts, pt_size=0.01, ns='visualizer_contacts', dt=rospy.Duration(5.0)):
  viz= TSimpleVisualizer(dt, name_space=ns)
  viz.AddContacts(contacts, scale=[pt_size])



if __name__=='__main__':
  import time
  rospy.init_node('viz_test')
  viz= TSimpleVisualizer(rospy.Duration(5.0))
  time.sleep(1.0)
  viz.viz_frame= 'base'
  x= [0.72, -0.91, 0.32,  0.27, 0.65, -0.27, 0.66]
  viz.AddCube(x, scale=[0.05,0.04,0.03], alpha=1.0, rgb=viz.ICol(5))
  x= [x[0]+0.1]+x[1:]
  viz.AddCoord(x, scale=[0.05,0.002], alpha=1.0)
  x= [x[0]+0.1]+x[1:]
  viz.AddMarker(x, scale=[0.05,0.05,0.008], alpha=1.0, rgb=viz.ICol(3))
  viz.AddArrow(x, scale=[0.05,0.002,0.002], alpha=1.0, rgb=viz.ICol(3))
