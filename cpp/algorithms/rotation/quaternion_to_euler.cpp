//-------------------------------------------------------------------------------------------
/*! \file    quaternion_to_euler.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.09, 2015

    ref.
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/

    Compare the result with:
    ../python/quaternion_to_euler.py
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
#include <cmath>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

// Compute YZX-Euler angles from quaternion.
// ref. http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
void QuaternionToYZXEuler(
    const double &qx, const double &qy, const double &qz, const double &qw,
    double &ey, double &ez, double &ex)
{
  double sqw = qw*qw;
  double sqx = qx*qx;
  double sqy = qy*qy;
  double sqz = qz*qz;
  double unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
  double test = qx*qy + qz*qw;
  if (test > 0.499*unit) { // singularity at north pole
          ey = 2.0 * std::atan2(qx,qw);
          ez = M_PI/2.0;
          ex = 0.0;
          return;
  }
  if (test < -0.499*unit) { // singularity at south pole
          ey = -2.0 * std::atan2(qx,qw);
          ez = -M_PI/2.0;
          ex = 0.0;
          return;
  }
  ey = std::atan2(2.0*qy*qw-2.0*qx*qz , sqx - sqy - sqz + sqw);
  ez = std::asin(2.0*test/unit);
  ex = std::atan2(2.0*qx*qw-2.0*qy*qz , -sqx + sqy - sqz + sqw);
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  double qx(0.0),qy(0.0),qz(0.0),qw(0.0);
  if(argc>1)  qx= atof(argv[1]);
  if(argc>2)  qy= atof(argv[2]);
  if(argc>3)  qz= atof(argv[3]);
  if(argc>4)  qw= atof(argv[4]);
  double ey(0.0),ez(0.0),ex(0.0);
  QuaternionToYZXEuler(qx,qy,qz,qw, ey,ez,ex);
  cout<<"Input quaternion: "<<qx<<", "<<qy<<", "<<qz<<", "<<qw<<std::endl;
  cout<<"YZX Euler angles: "<<ey<<", "<<ez<<", "<<ex<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
