#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}
template<typename T>
inline T URand(const T &min,const T &max)
{
  return (max-min)*URand<T>()+min;
}
template<typename T>
inline T NRand()
{
  T r(0.0l);
  for(int i(0);i<12;++i) r+=URand<T>();
  return r-6.0l;
}
using namespace std;
int func(const double &x1,const double &x2)
{
  if (x1*x1+0.2*x1*x2+2*x2*x2 < 1) return -1;
  return +1;
}
int main(int,char**)
{
  ofstream osi("d/input3.dat"),oso("d/output3.dat");
  for(int i(0); i<50; ++i)
  {
    double x1(0.5+0.5*NRand<double>()),x2(0.5+0.5*NRand<double>());
    double y(func(x1,x2));
    osi<<x1<<" "<<x2<<endl;
    oso<<y<<endl;
    cout<<x1<<" "<<x2<<" "<<y<<endl;
  }
  for(int i(0); i<20; ++i)
  {
    double x1(-0.6+0.3*NRand<double>()),x2(-0.6+0.3*NRand<double>());
    double y(func(x1,x2));
    osi<<x1<<" "<<x2<<endl;
    oso<<y<<endl;
    cout<<x1<<" "<<x2<<" "<<y<<endl;
  }
  return 0;
}
