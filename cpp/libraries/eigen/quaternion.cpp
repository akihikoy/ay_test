//-------------------------------------------------------------------------------------------
/*! \file    quaternion.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.18, 2015
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
//-------------------------------------------------------------------------------------------
using namespace std;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream &lhs, const Eigen::Quaterniond &rhs)
{
  lhs<<rhs.x()<<" "<<rhs.y()<<" "<<rhs.z()<<" "<<rhs.w();
  return lhs;
}

int main(int argc, char**argv)
{
  Eigen::Quaterniond q(Eigen::Quaterniond::Identity());
  print(q);
  Eigen::AngleAxisd rot(M_PI/6.0, Eigen::Vector3d::UnitY());
  q= rot*q;
  print(q);
  q= Eigen::AngleAxisd(5.0*M_PI/6.0, Eigen::Vector3d::UnitY()) * q;
  print(q);
  q= Eigen::AngleAxisd(M_PI/6.0, Eigen::Vector3d::UnitY()) * q;
  print(q);
  q= Eigen::AngleAxisd(5.0*M_PI/6.0, Eigen::Vector3d::UnitY()) * q;
  print(q);
  return 0;
}
//-------------------------------------------------------------------------------------------
