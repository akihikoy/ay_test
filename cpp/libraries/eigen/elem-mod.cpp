#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl
using namespace Eigen;
using namespace std;

// matlab-like mod function that returns always positive
template<typename T>
inline T Mod(const T &x, const T &y)
{
  if(y==0)  return x;
  return x-y*std::floor(x/y);
}
// convert radian to [-pi,pi)
double RadToNPiPPi(const double &x)
{
  return Mod(x+M_PI,M_PI*2.0)-M_PI;
}

int main()
{
  VectorXd y(2);
  for(Vector2d x(-10,10);x(0)<=10;x+=Vector2d(0.1,-0.1))
  {
    y= x.unaryExpr(&RadToNPiPPi);
    cout<<x.transpose()<<"  "<<y.transpose()<<endl;
  }
}
