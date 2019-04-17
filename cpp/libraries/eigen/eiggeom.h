//-------------------------------------------------------------------------------------------
/*! \file    eiggeom.h
    \brief   Eigen-based geometry library.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.03, 2017

src. testl/eigen/eiggeom.h
*/
//-------------------------------------------------------------------------------------------
#ifndef eiggeom_h
#define eiggeom_h
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

typedef double TEScalar;
typedef Eigen::Matrix<TEScalar,3,1> TEVector3;
typedef Eigen::Matrix<TEScalar,4,1> TEVector4;
typedef Eigen::Matrix<TEScalar,6,1> TEVector6;
typedef Eigen::Matrix<TEScalar,7,1> TEVector7;
typedef Eigen::Matrix<TEScalar,3,3> TEMatrix3;
typedef Eigen::Quaternion<TEScalar> TEQuaternion;
typedef Eigen::Translation<TEScalar,3> TETranslation;

template <typename t_iterator>
inline TEVector7 ToEVec(t_iterator itr)
{
  TEVector7 x;
  for(int i(0);i<7;++i,++itr)  x[i]=*itr;
  return x;
  // return Eigen::Map<TEVector7>(itr);
}
template <typename t_iterator>
inline TEVector3 ToEVec3(t_iterator itr)
{
  TEVector3 x;
  for(int i(0);i<3;++i,++itr)  x[i]=*itr;
  return x;
}
//-------------------------------------------------------------------------------------------

// Pose x=[x,y,z,quaternion(qx,qy,qz,qw)] to Quaternion
inline TEQuaternion XToQ(const TEVector7 &x)
{
  return TEQuaternion(x[6],x[3],x[4],x[5]);
}
//-------------------------------------------------------------------------------------------

// Pose x=[x,y,z,quaternion(qx,qy,qz,qw)] or position x=[x,y,z] to Translation
template <typename t_vector>
inline TETranslation XToPos(const t_vector &x)
{
  return TETranslation(x[0],x[1],x[2]);
}
//-------------------------------------------------------------------------------------------

// Position [x,y,z] and Quaternion to pose x=[x,y,z,quaternion(qx,qy,qz,qw)]
inline TEVector7 PosQToX(const TEVector3 &p, const TEQuaternion &q)
{
  TEVector7 x;
  x << p[0],p[1],p[2], q.x(),q.y(),q.z(),q.w();
  return x;
}
//-------------------------------------------------------------------------------------------

// Quaternion to 3x3 rotation matrix
inline TEMatrix3 QToRot(const TEQuaternion &q)
{
  return q.matrix();
}
//-------------------------------------------------------------------------------------------

inline TEQuaternion QFromAxisAngle(const TEVector3 &axis, const TEScalar &angle)
{
  return TEQuaternion(Eigen::AngleAxis<TEScalar>(angle, axis));
}
//-------------------------------------------------------------------------------------------

// Compute "x2 * x1"; x1,x2 are [x,y,z,quaternion]
inline TEVector7 Transform(const TEVector7 &x2, const TEVector7 &x1)
{
  TEVector3 p= ((XToQ(x2)*XToPos(x1)) * XToPos(x2)).translation();
  TEQuaternion q= XToQ(x2)*XToQ(x1);
  return PosQToX(p,q);
}
inline TEVector7 Transform(const TEQuaternion &q2, const TEVector7 &x1)
{
  TEVector3 p= (q2*XToPos(x1)).translation();
  TEQuaternion q= q2*XToQ(x1);
  return PosQToX(p,q);
}
inline TEVector7 Transform(const TEVector3 &p2, const TEVector7 &x1)
{
  TEVector3 p= (TETranslation(p2) * XToPos(x1)).translation();
  TEQuaternion q= XToQ(x1);
  return PosQToX(p,q);
}
inline TEVector3 Transform(const TEVector7 &x2, const TEVector3 &x1)
{
  // TEVector3 p= ((XToQ(x2)*XToPos(x1)) * XToPos(x2)).translation();
  // TEVector3 p= (XToPos(x2) * (XToQ(x2)*XToPos(x1))).translation();
  TEVector3 p= (XToPos((XToQ(x2)*XToPos(x1)).translation()) * XToPos(x2)).translation();
  return p;
}
//-------------------------------------------------------------------------------------------

// This solves for x in "x_r = x_l * x", i.e. return "inv(x_l)*x_r"
// For example, get a local pose of x_r in the x_l frame
// x_* are [x,y,z,quaternion] form
inline TEVector7 TransformLeftInv(const TEVector7 &x_l, const TEVector7 &x_r)
{
  TEVector3 p= (XToQ(x_l).inverse() * (XToPos(x_r)*XToPos(x_l).inverse())).translation();
  TEQuaternion q= XToQ(x_l).inverse() * XToQ(x_r);
  return PosQToX(p,q);
}
// Assuming x_r = [x,y,z].
inline TEVector3 TransformLeftInv(const TEVector7 &x_l, const TEVector3 &x_r)
{
  TEVector3 p= (XToQ(x_l).inverse() * (XToPos(x_r)*XToPos(x_l).inverse())).translation();
  return p;
}
//-------------------------------------------------------------------------------------------

// This solves for trans_x in "x_l = trans_x * x_r", i.e. return "x_l*inv(x_r)"
// For example, get a transformation, x_r to x_l
// x_* are [x,y,z,quaternion] form
inline TEVector7 TransformRightInv(const TEVector7 &x_l, const TEVector7 &x_r)
{
  TEQuaternion qt= XToQ(x_l) * XToQ(x_r).inverse();
  TEVector3 pt= (XToPos(x_l) * qt * XToPos(x_r).inverse()).translation();
  return PosQToX(pt,qt);
}
//-------------------------------------------------------------------------------------------

inline TEVector3 InvRodrigues(const TEMatrix3 &R, const TEScalar &epsilon=1.0e-6)
{
  TEScalar alpha= (R(0,0)+R(1,1)+R(2,2) - 1.0) / 2.0;

  if((alpha-1.0 < epsilon) && (alpha-1.0 > -epsilon))
  {
    return TEVector3(0.0,0.0,0.0);
  }
  else
  {
    TEVector3 w(0.0,0.0,0.0);
    TEScalar th= std::acos(alpha);
    TEScalar tmp= 0.5 * th / std::sin(th);
    w[0] = tmp * (R(2,1) - R(1,2));
    w[1] = tmp * (R(0,2) - R(2,0));
    w[2] = tmp * (R(1,0) - R(0,1));
    return w;
  }
}
//-------------------------------------------------------------------------------------------

// Get difference of two poses [dx,dy,dz, dwx,dwy,dwz] (like x2-x1)
inline TEVector6 DiffX(const TEVector7 &x1, const TEVector7 &x2)
{
  TEVector3 w= InvRodrigues(QToRot(XToQ(x2))*QToRot(XToQ(x1).inverse()));
  TEVector6 dx;
  dx << x2[0]-x1[0],x2[1]-x1[1],x2[2]-x1[2], w[0],w[1],w[2];
  return dx;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // eiggeom_h
//-------------------------------------------------------------------------------------------
