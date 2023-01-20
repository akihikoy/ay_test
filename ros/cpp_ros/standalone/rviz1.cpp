//-------------------------------------------------------------------------------------------
/*! \file    rviz1.cpp
    \brief   Test of RViz utility.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Dec.15, 2022

g++ -O2 -g -W -Wall -o rviz1.out rviz1.cpp -I/usr/include/eigen3 -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/ColorRGBA.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <set>
#include <string>
//-------------------------------------------------------------------------------------------
namespace trick
{

// Copied from ay_cpp.geom_util.h

// (Ported from ay_py.core.geom)
// Orthogonalize a vector vec w.r.t. base; i.e. vec is modified so that dot(vec,base)==0.
// original_norm: keep original vec's norm, otherwise the norm is 1.
// Using The Gram-Schmidt process: http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
template <typename t_value>
inline void Orthogonalize(const t_value vec[3], const t_value base[3], t_value out[3], bool original_norm=true)
{
  typedef Eigen::Matrix<t_value,3,1> Vector3;
  Vector3 v(vec);
  Vector3 vbase= Vector3(base).normalized();
  Vector3 v2= v - v.dot(vbase)*vbase;
  Eigen::Map<Vector3> vout(out);
  vout= (original_norm ? (v2.normalized()*v.norm()) : v2.normalized());
}
//-------------------------------------------------------------------------------------------

// (Ported from ay_py.core.geom)
// Get an orthogonal axis of a given axis
// preferable: preferable axis (orthogonal axis is close to this)
// fault: return this axis when dot(axis,preferable)==1
template <typename t_value>
inline void GetOrthogonalAxisOf(const t_value axis[3], t_value out[3], const t_value preferable[3]=NULL, const t_value fault[]=NULL)
{
  static const t_value default_preferable[3]= {0.0,0.0,1.0};
  if(preferable==NULL)  preferable= default_preferable;
  typedef Eigen::Matrix<t_value,3,1> Vector3;
  t_value naxis[3];
  Eigen::Map<Vector3> vnaxis(naxis);
  vnaxis= Vector3(axis).normalized();

  if(fault==NULL || 1.0-std::fabs(vnaxis.dot(Vector3(preferable)))>=1.0e-6)
    Orthogonalize(preferable, /*base=*/naxis, out, /*original_norm=*/false);
  else
  {
    Eigen::Map<Vector3> vout(out);
    vout= Vector3(fault);
  }
}
//-------------------------------------------------------------------------------------------

// (Ported from ay_py.core.geom)
// For visualizing cylinder, arrow, etc., get a pose x from two points p1-->p2.
// Axis ax decides which axis corresponds to p1-->p2.
// Ratio r decides: r=0: x is on p1, r=1: x is on p2, r=0.5: x is on the middle of p1 and p2.
template <typename t_value>
inline void XFromP1P2(const t_value p1[3], const t_value p2[3], t_value x_out[7], char ax='z', const t_value &r=0.5)
{
  typedef Eigen::Matrix<t_value,3,1> Vector3;
  typedef Eigen::Matrix<t_value,4,1> Vector4;
  typedef Eigen::Matrix<t_value,7,1> Vector7;
  Vector3 v1(p1), v2(p2);
  t_value aex[3],aey[3],aez[3];
  Eigen::Map<Vector3> ex(aex),ey(aey),ez(aez);
  if(ax=='x')
  {
    ex= (v2-v1).normalized();
    t_value preferable[]={0.0,1.0,0.0}, fault[]={0.0,0.0,1.0};
    GetOrthogonalAxisOf(aex, /*out=*/aey, preferable, fault);
    ez= ex.cross(ey);
  }
  else if(ax=='y')
  {
    ey= (v2-v1).normalized();
    t_value preferable[]={0.0,0.0,1.0}, fault[]={1.0,0.0,0.0};
    GetOrthogonalAxisOf(aey, /*out=*/aez, preferable, fault);
    ex= ey.cross(ez);
  }
  else if(ax=='z')
  {
    ez= (v2-v1).normalized();
    t_value preferable[]={1.0,0.0,0.0}, fault[]={0.0,1.0,0.0};
    GetOrthogonalAxisOf(aez, /*out=*/aex, preferable, fault);
    ey= ez.cross(ex);
  }
  Eigen::Map<Vector7> x(x_out);
  x.segment(0,3)= (1.0-r)*v1+r*v2;
  Eigen::Matrix<t_value,3,3> rot;
  rot<<ex,ey,ez;
  Eigen::Quaternion<t_value> q(rot);
  x.segment(3,4)= Vector4(q.x(),q.y(),q.z(),q.w());
}
//-------------------------------------------------------------------------------------------


// Convert p to geometry_msgs/Point; usually, t_point==geometry_msgs::Point
template <typename t_array, typename t_point>
inline void PToGPoint(const t_array p, t_point &point)
{
  point.x= p[0];
  point.y= p[1];
  point.z= p[2];
}
// Convert p to geometry_msgs/Point; usually, t_point==geometry_msgs::Point
template <typename t_point, typename t_array>
inline t_point PToGPoint(const t_array p)
{
  t_point point;
  PToGPoint<t_array,t_point>(p, point);
  return point;
}
//-------------------------------------------------------------------------------------------

// Convert geometry_msgs/Point to p; usually, t_point==geometry_msgs::Point
template <typename t_point, typename t_array>
inline void GPointToP(const t_point &point, t_array p)
{
  p[0]= point.x;
  p[1]= point.y;
  p[2]= point.z;
}
//-------------------------------------------------------------------------------------------

// Convert x to geometry_msgs/Pose; usually, t_pose==geometry_msgs::Pose
template <typename t_array, typename t_pose>
inline void XToGPose(const t_array x, t_pose &pose)
{
  pose.position.x= x[0];
  pose.position.y= x[1];
  pose.position.z= x[2];
  pose.orientation.x= x[3];
  pose.orientation.y= x[4];
  pose.orientation.z= x[5];
  pose.orientation.w= x[6];
}
// Convert x to geometry_msgs/Pose; usually, t_pose==geometry_msgs::Pose
template <typename t_pose, typename t_array>
inline t_pose XToGPose(const t_array x)
{
  t_pose pose;
  XToGPose<t_array,t_pose>(x, pose);
  return pose;
}
//-------------------------------------------------------------------------------------------

// Convert geometry_msgs/Pose to x; usually, t_pose==geometry_msgs::Pose
template <typename t_pose, typename t_array>
inline void GPoseToX(const t_pose &pose, t_array x)
{
  x[0]= pose.position.x;
  x[1]= pose.position.y;
  x[2]= pose.position.z;
  x[3]= pose.orientation.x;
  x[4]= pose.orientation.y;
  x[5]= pose.orientation.z;
  x[6]= pose.orientation.w;
}
//-------------------------------------------------------------------------------------------

// t_point: e.g. geometry_msgs::Point, Vector3
template<typename t_point, typename t_value=double>
inline t_point GenGPoint(const t_value &x=0.0, const t_value &y=0.0, const t_value &z=0.0)
{
  t_point p;
  p.x= x;
  p.y= y;
  p.z= z;
  return p;
}
//-------------------------------------------------------------------------------------------
// t_quaternion: e.g. geometry_msgs::Quaternion
template<typename t_quaternion, typename t_value=double>
inline t_quaternion GenGQuaternion(const t_value &x=0.0, const t_value &y=0.0, const t_value &z=0.0, const t_value &w=0.0)
{
  t_quaternion p;
  p.x= x;
  p.y= y;
  p.z= z;
  p.w= w;
  return p;
}
//-------------------------------------------------------------------------------------------
// t_rgba: e.g. geometry_msgs::ColorRGBA
template<typename t_rgba, typename t_value_r=double, typename t_value_g=double, typename t_value_b=double, typename t_value_a=double>
inline t_rgba GenGRBGA(const t_value_r &r=1.0, const t_value_g &g=1.0, const t_value_b &b=1.0, const t_value_a &a=1.0)
{
  t_rgba c;
  c.r= r;
  c.g= g;
  c.b= b;
  c.a= a;
  return c;
}
//-------------------------------------------------------------------------------------------








}
//-------------------------------------------------------------------------------------------

using namespace trick;


// Utility for RViz.
class TSimpleVisualizer
{
protected:
  ros::Publisher viz_pub_;
  int curr_id_;
  std::set<int> added_ids_;
  std::string viz_frame_;
  std::string viz_ns_;
  ros::Duration viz_dt_;

  virtual void marker_operation(const visualization_msgs::Marker &marker)
    {
      viz_pub_.publish(marker);
    }

public:
  typedef std_msgs::ColorRGBA ColorRGBA;
  typedef geometry_msgs::Vector3 Vector3;
  typedef geometry_msgs::Pose Pose;
  typedef geometry_msgs::Point Point;
  typedef geometry_msgs::Quaternion Quaternion;
  typedef visualization_msgs::Marker Marker;
  typedef visualization_msgs::MarkerArray MarkerArray;

  TSimpleVisualizer(const ros::Duration &viz_dt=ros::Duration(), const std::string &name_space="visualizer",
               const std::string &frame="", int queue_size=1, const std::string &topic="visualization_marker")
    {
      ros::NodeHandle node;
      if(topic!="")
        viz_pub_= node.advertise<Marker>(topic, queue_size);
      curr_id_= 0;
      added_ids_.clear();
      viz_frame_= frame==""?"base":frame;
      viz_ns_= name_space;
      viz_dt_= viz_dt;
    }

  ~TSimpleVisualizer()
    {
      if(viz_dt_!=ros::Duration())
        DeleteAllMarkers();
      Reset();
      viz_pub_.shutdown();
    }

  virtual void Reset()
    {
      curr_id_= 0;
    }

  virtual void DeleteMarker(int mid)
    {
      Marker marker;
      marker.header.frame_id= viz_frame_;
      marker.ns= viz_ns_;
      marker.id= mid;
      marker.action= Marker::DELETE;
      marker_operation(marker);
      added_ids_.erase(mid);
    }

  virtual void DeleteAllMarkers()
    {
      Marker marker;
      marker.header.frame_id= viz_frame_;
      marker.ns= viz_ns_;
      marker.action= Marker::DELETEALL;
      marker_operation(marker);
      added_ids_.clear();
    }

  ColorRGBA ICol(int i) const
    {
      i= i%7;
      switch(i)
      {
        case 0:  return GenGRBGA<ColorRGBA>(1,0,0, 1);
        case 1:  return GenGRBGA<ColorRGBA>(0,1,0, 1);
        case 2:  return GenGRBGA<ColorRGBA>(0,0,1, 1);
        case 3:  return GenGRBGA<ColorRGBA>(1,1,0, 1);
        case 4:  return GenGRBGA<ColorRGBA>(1,0,1, 1);
        case 5:  return GenGRBGA<ColorRGBA>(0,1,1, 1);
        case 6:  return GenGRBGA<ColorRGBA>(1,1,1, 1);
      }
      return GenGRBGA<ColorRGBA>();
    }

  Marker GenMarker(const Pose &x, const Vector3 &scale, const ColorRGBA &rgb, const float &alpha) const
    {
      Marker marker;
      marker.header.frame_id= viz_frame_;
      marker.header.stamp= ros::Time::now();
      marker.ns= viz_ns_;
      marker.action= Marker::ADD;
      marker.lifetime= viz_dt_;
      marker.scale= scale;
      marker.color= GenGRBGA<ColorRGBA>(rgb.r,rgb.g,rgb.b,alpha);
      marker.pose= x;
      return marker;
    }

  int SetID(Marker &marker, int mid)
    {
      if(mid<0)
      {
        marker.id= curr_id_;
        curr_id_++;
      }
      else
      {
        marker.id= mid;
        if(marker.id>=curr_id_)
          curr_id_= marker.id+1;
      }
      added_ids_.insert(marker.id);
      return marker.id+1;
    }

//   #Visualize a marker at x.  If mid is None, the id is automatically assigned
//   def AddMarker(self, x, scale=[0.02,0.02,0.004], rgb=[1,1,1], alpha=1.0, mid=None):
//     marker= self.GenMarker(x, scale, rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.CUBE  # or CUBE, SPHERE, ARROW, CYLINDER
//     marker_operation(marker)
//     return mid2

  // Visualize an arrow at x.  If mid is None, the id is automatically assigned
  int AddArrow(const Pose &x, const Vector3 &scale=GenGPoint<Vector3>(0.05,0.002,0.002), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::ARROW;
      marker_operation(marker);
      return mid2;
    }

//   #Visualize an arrow from p1 to p2.  If mid is None, the id is automatically assigned
//   def AddArrowP2P(self, p1, p2, diameter=0.002, rgb=[1,1,1], alpha=1.0, mid=None):
//     x= XFromP1P2(p1, p2, ax='x', r=0)
//     length= la.norm(Vec(p2)-Vec(p1))
//     scale= [length,diameter,diameter]
//     marker= self.GenMarker(x, scale, rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.ARROW  # or CUBE, SPHERE, ARROW, CYLINDER
//     marker_operation(marker)
//     return mid2

//   #Visualize a list of arrows.  If mid is None, the id is automatically assigned
//   def AddArrowList(self, x_list, axis='x', scale=[0.05,0.002], rgb=[1,1,1], alpha=1.0, mid=None):
//     iex,iey= {'x':(0,1),'y':(1,2),'z':(2,0)}[axis]
//     def point_on_arrow(x,l):
//       exyz= RotToExyz(QToRot(x[3:]))
//       ex,ey= exyz[iex],exyz[iey]
//       pt= x[:3]+l*ex
//       return [x[:3], pt, pt, x[:3]+0.7*l*ex+0.15*l*ey, pt, x[:3]+0.7*l*ex-0.15*l*ey]
//     x= [0,0,0, 0,0,0,1]
//     if not isinstance(rgb[0],(int,float)):  rgb= np.repeat(rgb,6,axis=0)
//     marker= self.GenMarker(x, [scale[1],0.0,0.0], rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.LINE_LIST
//     marker.points= [geometry_msgs.msg.Point(*p) for x in x_list for p in point_on_arrow(x,scale[0])]
//     marker_operation(marker)
//     return mid2

  // Visualize a cube at x.  If mid is None, the id is automatically assigned
  int AddCube(const Pose &x, const Vector3 &scale=GenGPoint<Vector3>(0.05,0.03,0.03), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::CUBE;
      marker_operation(marker);
      return mid2;
    }

//   #Visualize a list of cubes [[x,y,z]*N].  If mid is None, the id is automatically assigned
//   def AddCubeList(self, points, scale=[0.05,0.03,0.03], rgb=[1,1,1], alpha=1.0, mid=None):
//     x= [0,0,0, 0,0,0,1]
//     marker= self.GenMarker(x, scale, rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.CUBE_LIST
//     marker.points= [geometry_msgs.msg.Point(*p[:3]) for p in points]
//     marker_operation(marker)
//     return mid2

  // Visualize a sphere at x.  If mid is None, the id is automatically assigned
  int AddSphere(const Pose &x, const Vector3 &scale=GenGPoint<Vector3>(0.05,0.05,0.05), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::SPHERE;
      marker_operation(marker);
      return mid2;
    }
  // Visualize a sphere at p=[x,y,z].  If mid is None, the id is automatically assigned
  int AddSphere(const Point &p, const Vector3 &scale=GenGPoint<Vector3>(0.05,0.05,0.05), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Pose x;
      x.position= p;
      x.orientation= GenGQuaternion<Quaternion>(0.,0.,0.,1.);
      return AddSphere(x, scale, rgb, alpha, mid);
    }

//   #Visualize a list of spheres [[x,y,z]*N].  If mid is None, the id is automatically assigned
//   def AddSphereList(self, points, scale=[0.05,0.05,0.05], rgb=[1,1,1], alpha=1.0, mid=None):
//     x= [0,0,0, 0,0,0,1]
//     marker= self.GenMarker(x, scale, rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.SPHERE_LIST
//     marker.points= [geometry_msgs.msg.Point(*p[:3]) for p in points]
//     marker_operation(marker)
//     return mid2

  // Visualize a cylinder whose end points are p1 and p2.  If mid is None, the id is automatically assigned
  int AddCylinder(const Point &p1, const Point &p2, const float &diameter, const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      typedef Eigen::Matrix<float,3,1> EVec3;
      float ap1[3], ap2[3], pose[7];
      GPointToP(p1, ap1);
      GPointToP(p2, ap2);
      XFromP1P2(ap1, ap2, /*x_out=*/pose, /*ax=*/'z', /*r=*/0.5f);
      Pose x;
      XToGPose(pose, x);
      float length= (EVec3(ap2)-EVec3(ap1)).norm();

      Vector3 scale= GenGPoint<Vector3>(diameter,diameter,length);
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::CYLINDER;
      marker_operation(marker);
      return mid2;
    }

//   #Visualize a cylinder at x along its axis ('x','y','z').  If mid is None, the id is automatically assigned
//   def AddCylinderX(self, x, axis, diameter, l1, l2, rgb=[1,1,1], alpha=1.0, mid=None):
//     e= RotToExyz(QToRot(x[3:]))[{'x':0,'y':1,'z':2}[axis]]
//     p1= x[:3]+l1*e
//     p2= x[:3]+l2*e
//     return self.AddCylinder(p1,p2, diameter, rgb=rgb, alpha=alpha, mid=mid)

  // Visualize a points [[x,y,z]*N].  If mid is None, the id is automatically assigned
  int AddPoints(const std::vector<Point> &points, const Vector3 &scale=GenGPoint<Vector3>(0.03,0.03), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Pose x;
      x.position= GenGPoint<Point>(0.,0.,0.);
      x.orientation= GenGQuaternion<Quaternion>(0.,0.,0.,1.);
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::POINTS;
      marker.points= points;
      marker_operation(marker);
      return mid2;
    }

//   #Visualize a coordinate system at x with arrows.  If mid is None, the id is automatically assigned
//   def AddCoord(self, x, scale=[0.05,0.002], alpha=1.0, mid=None):
//     scale= [scale[0],scale[1],scale[1]]
//     p,R= XToPosRot(x)
//     Ry= np.array([R[:,1],R[:,2],R[:,0]]).T
//     Rz= np.array([R[:,2],R[:,0],R[:,1]]).T
//     mid= self.AddArrow(x, scale=scale, rgb=self.ICol(0), alpha=alpha, mid=mid)
//     mid= self.AddArrow(PosRotToX(p,Ry), scale=scale, rgb=self.ICol(1), alpha=alpha, mid=mid)
//     mid= self.AddArrow(PosRotToX(p,Rz), scale=scale, rgb=self.ICol(2), alpha=alpha, mid=mid)
//     return mid

//   #Visualize a coordinate system at x with cylinders.  If mid is None, the id is automatically assigned
//   def AddCoordC(self, x, scale=[0.05,0.002], alpha=1.0, mid=None):
//     scale= [scale[0],scale[1],scale[1]]
//     p,R= XToPosRot(x)
//     Ry= np.array([R[:,1],R[:,2],R[:,0]]).T
//     Rz= np.array([R[:,2],R[:,0],R[:,1]]).T
//     mid= self.AddCylinderX(x, 'x', scale[1], 0, scale[0], rgb=self.ICol(0), alpha=alpha, mid=mid)
//     mid= self.AddCylinderX(x, 'y', scale[1], 0, scale[0], rgb=self.ICol(1), alpha=alpha, mid=mid)
//     mid= self.AddCylinderX(x, 'z', scale[1], 0, scale[0], rgb=self.ICol(2), alpha=alpha, mid=mid)
//     return mid

  // Visualize a polygon [[x,y,z]*N].  If mid is None, the id is automatically assigned
  int AddPolygon(const std::vector<Point> &points, const Vector3 &scale=GenGPoint<Vector3>(0.02), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Pose x;
      x.position= GenGPoint<Point>(0.,0.,0.);
      x.orientation= GenGQuaternion<Quaternion>(0.,0.,0.,1.);
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::LINE_STRIP;
      marker.points= points;
      marker_operation(marker);
      return mid2;
    }

  // Visualize a list of lines [[x,y,z]*N] (2i-th and (2i+1)-th points are pair).  If mid is None, the id is automatically assigned
  int AddLineList(const std::vector<Point> &points, const Vector3 &scale=GenGPoint<Vector3>(0.02), const ColorRGBA &rgb=GenGRBGA<ColorRGBA>(), const float &alpha=1.0, int mid=-1)
    {
      Pose x;
      x.position= GenGPoint<Point>(0.,0.,0.);
      x.orientation= GenGQuaternion<Quaternion>(0.,0.,0.,1.);
      Marker marker= GenMarker(x, scale, rgb, alpha);
      int mid2= SetID(marker,mid);
      marker.type= Marker::LINE_LIST;
      marker.points= points;
      marker_operation(marker);
      return mid2;
    }

//   #Visualize a text.  If mid is None, the id is automatically assigned
//   def AddText(self, p, text, scale=[0.02], rgb=[1,1,1], alpha=1.0, mid=None):
//     if len(p)==3:
//       x= list(p)+[0,0,0,1]
//     else:
//       x= p
//     marker= self.GenMarker(x, [0.0,0.0]+list(scale), rgb, alpha)
//     mid2= self.SetID(marker,mid)
//     marker.type= visualization_msgs.msg.Marker.TEXT_VIEW_FACING
//     marker.text= text
//     marker_operation(marker)
//     return mid2

//   #Visualize contacts which should be an moveit_msgs/ContactInformation[] [DEPRECATED:arm_navigation_msgs/ContactInformation[]]
//   def AddContacts(self, contacts, with_normal=False, scale=[0.01], rgb=[1,1,0], alpha=0.7, mid=None):
//     if len(contacts)==0:  return curr_id_
//     cscale= scale*3
//     viz_frame_= contacts[0].header.frame_id
//     for c in contacts:
//       p= [c.position.x, c.position.y, c.position.z]
//       mid= self.AddSphere(p+[0.,0.,0.,1.], scale=cscale, rgb=rgb, alpha=alpha, mid=mid)
//       if with_normal:
//         x= XFromP1P2(p, Vec(p)+Vec([c.normal.x, c.normal.y, c.normal.z]), ax='x', r=0.0)
//         ascale= [c.depth,0.2*scale[0],0.2*scale[0]]
//         mid= self.AddArrow(x, scale=ascale, rgb=rgb, alpha=alpha, mid=mid)
//     return mid

};
//-------------------------------------------------------------------------------------------

// Utility for RViz (MarkerArray version of TSimpleVisualizer).
class TSimpleVisualizerArray : public TSimpleVisualizer
{
protected:
  MarkerArray marker_array_;

  /*override*/void marker_operation(const Marker &marker)
    {
      marker_array_.markers.push_back(marker);
    }

public:
  TSimpleVisualizerArray(const ros::Duration &viz_dt=ros::Duration(), const std::string &name_space="visualizer",
                         const std::string &frame="", int queue_size=1, const std::string &topic="visualization_marker_array")
      : TSimpleVisualizer(viz_dt, name_space, frame, queue_size, /*topic=*/"")
    {
      ros::NodeHandle node;
      if(topic!="")
        viz_pub_= node.advertise<MarkerArray>(topic, queue_size);
      Reset();
    }

  /*override*/void Reset()
    {
      curr_id_= 0;
      marker_array_= MarkerArray();
    }

  void Publish()
    {
      viz_pub_.publish(marker_array_);
      Reset();
    }

  /*override*/void DeleteAllMarkers()
    {
      Reset();
      Marker marker;
      marker.header.frame_id= viz_frame_;
      marker.ns= viz_ns_;
      marker.action= Marker::DELETEALL;
      marker_operation(marker);
      Publish();
      added_ids_.clear();
    }
};
//-------------------------------------------------------------------------------------------




using namespace std_msgs;  // std_msgs::ColorRGBA
using namespace geometry_msgs;  // geometry_msgs::Vector3, Pose, Point, Quaternion
using namespace visualization_msgs;  // visualization_msgs::Marker, visualization_msgs::MarkerArray

Point TestPointAt(const float &t, const float &z, const float &dt=0.0f, const float &x=1.0)
{
  return GenGPoint<Point,float>(x,0.5*std::sin(t+dt),z);
}
Pose TestPoseAt(const float &t, const float &z, const float &dt=0.0f, const float &x=1.0)
{
  Pose pose;
  pose.position= TestPointAt(t,z,dt,x);
  pose.orientation= GenGQuaternion<Quaternion>(0.,0.,0.,1.);
  return pose;
}
std::vector<Point> TestPointsAt(const float &t, const float &z, const float &dt=0.0f, const float &x=1.0)
{
  std::vector<Point> points;
  for(int i(0);i<10;++i)  points.push_back(TestPointAt(t,z+0.05*std::cos(float(i)*6.28/10.+dt),dt,x+0.1*std::sin(float(i)*6.28/10.+dt)));
  return points;
}

int main(int argc, char**argv)
{
  ros::init(argc, argv, "rviz1");
  TSimpleVisualizer viz(ros::Duration(0.1),/*name_space=*/"viz", /*frame=*/"", /*queue_size=*/10);
  TSimpleVisualizerArray viz_array(ros::Duration(0.1),/*name_space=*/"viz_array");
  float hz(20.), dz(0.1);
  ros::Rate rate_adjuster(hz);
  for(float t(0.0);ros::ok();t+=1./hz)
  {
    int c(0);
    int mid(0);
    mid= viz.AddArrow(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.01,0.01), viz.ICol(1), /*alpha=*/1.0, mid);
    mid= viz.AddCube(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.03,0.03), viz.ICol(1), /*alpha=*/1.0, mid);
    mid= viz.AddSphere(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.05,0.05), viz.ICol(1), /*alpha=*/1.0, mid);
    mid= viz.AddCylinder(TestPointAt(t,dz*c), TestPointAt(t,dz*c+0.05), /*diameter=*/0.1, viz.ICol(1), /*alpha=*/1.0, mid);
    ++c;
    mid= viz.AddPoints(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.03,0.03), viz.ICol(1), /*alpha=*/1.0, mid);
    mid= viz.AddPolygon(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.02), viz.ICol(1), /*alpha=*/1.0, mid);
    mid= viz.AddLineList(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.04), viz.ICol(1), /*alpha=*/1.0, mid);
    int mid2(0);
    mid2= viz_array.AddArrow(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.01,0.01), viz.ICol(2), /*alpha=*/1.0, mid2);
    mid2= viz_array.AddCube(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.03,0.03), viz.ICol(2), /*alpha=*/1.0, mid2);
    mid2= viz_array.AddSphere(TestPoseAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.05,0.05,0.05), viz.ICol(2), /*alpha=*/1.0, mid2);
    mid2= viz_array.AddCylinder(TestPointAt(t,dz*c), TestPointAt(t,dz*c,0.1), /*diameter=*/0.1, viz.ICol(2), /*alpha=*/1.0, mid2);
    ++c;
    mid2= viz_array.AddPoints(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.03,0.03), viz.ICol(2), /*alpha=*/1.0, mid2);
    mid2= viz_array.AddPolygon(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.02), viz.ICol(2), /*alpha=*/1.0, mid2);
    mid2= viz_array.AddLineList(TestPointsAt(t,dz*(++c)), /*scale=*/GenGPoint<Vector3>(0.04), viz.ICol(2), /*alpha=*/1.0, mid2);
    viz_array.Publish();
    rate_adjuster.sleep();
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
