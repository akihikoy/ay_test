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
double func(const double &x1,const double &x2)
{
  return x1*sin(2.0*M_PI*x2);
}
int main(int,char**)
{
  ofstream osi("d/input1.dat"),oso("d/output1.dat");
  for(int i(0); i<200; ++i)
  {
    double x1(URand(-1.0,0.0)),x2(URand(-1.0,0.0));
    double y(func(x1,x2)+0.1*NRand<double>());
    osi<<x1<<" "<<x2<<endl;
    oso<<y<<endl;
    cout<<x1<<" "<<x2<<" "<<y<<endl;
  }
  for(int i(0); i<200; ++i)
  {
    double x1(URand(0.0,1.0)),x2(URand(0.0,1.0));
    double y(func(x1,x2)+0.1*NRand<double>());
    osi<<x1<<" "<<x2<<endl;
    oso<<y<<endl;
    cout<<x1<<" "<<x2<<" "<<y<<endl;
  }
  return 0;
}
