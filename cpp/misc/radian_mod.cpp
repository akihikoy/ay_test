#include <cmath>
#include <iostream>

// matlab-like mod function that returns always positive
template<typename T>
inline T FMod(const T &x, const T &y)
{
  if(y==0)  return x;
  return x-y*std::floor(x/y);
}
// convert radian to [-pi,pi)
double RadToNPiPPi(const double &x)
{
  return FMod(x+M_PI,M_PI*2.0)-M_PI;
}

int main()
{
  using namespace std;
  for(double r(-10.0); r<10.0; r+=0.05)
  {
    double r2= RadToNPiPPi(r);
    cout<<r<<" "<<r2<<endl;
  }
  return 0;
}
