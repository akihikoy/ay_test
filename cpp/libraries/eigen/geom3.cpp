//-------------------------------------------------------------------------------------------
/*! \file    geom3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2017

compile:
  g++ -g -Wall geom3.cpp -I/usr/include/eigen3
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include "eiggeom.h"
//-------------------------------------------------------------------------------------------
namespace trick
{
}
//-------------------------------------------------------------------------------------------
#include <vector>
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  double _x10[]= {0.1,0.2,0.3, 0.41562693777745346, 0.41562693777745346, 0.0, 0.8090169943749475};
  std::vector<double> _x11(_x10,_x10+7);
  TEVector7 x1= ToEVec(_x11.begin());

  double _x20[]= {0.3,0.1,0.2, 0.6724985119639574, 0.6724985119639574, 0.0, 0.3090169943749474};
  std::vector<double> _x21(_x20,_x20+7);
  TEVector7 x2= ToEVec(_x21.begin());

  print(x1.transpose());
  print(x2.transpose());
  print(DiffX(x2,x1).transpose());

  TEVector7 x3= Transform(TEVector3(1.0,0.0,1.0), x2);
  TEVector7 x4= Transform(QFromAxisAngle(TEVector3(0.0,1.0,0.0),M_PI*0.2), x3);

  print(x3.transpose());
  print(x4.transpose());
  print(DiffX(x4,x2).transpose());
  print(DiffX(x4,x1).transpose());
  print(TransformLeftInv(x4,x2).transpose());
  print(TransformLeftInv(x3,x1).transpose());

  std::vector<double> _x12(_x10,_x10+3);
  TEVector3 p1= ToEVec3(_x12.begin());
  TEVector3 p2= Transform(x2,p1);
// print((XToPos(p1)*XToPos(p2)).translation().transpose());
// print((XToPos(p2)*XToPos(p1)).translation().transpose());
// print((XToPos(p1)*XToPos(p1)).translation().transpose());
// print(((XToQ(x2)*XToPos(p1))).translation().transpose());
// print((XToPos(x2) * (XToQ(x2)*XToPos(p1)) ).translation().transpose());
// print(((XToQ(x2)*XToPos(p1)) * XToPos(x2)).translation().transpose());
// print((XToPos((XToQ(x2)*XToPos(p1)).translation()) * XToPos(x2)).translation().transpose());
  print(p1.transpose());
  print(p2.transpose());
  print(TransformLeftInv(x2,p2).transpose());
  print(Transform(x2,TransformLeftInv(x2,p2)).transpose());

  return 0;
}
//-------------------------------------------------------------------------------------------
