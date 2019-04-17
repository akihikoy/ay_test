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
template<typename T>
inline T Sq(const T &x)
{
  return x*x;
}
using namespace std;
int main(int,char**)
{
  ofstream osi("d/input5.dat");
  for(int i(0); i<50; ++i)
  {
    double x1(0.2+0.2*NRand<double>()),x2(0.2+0.2*NRand<double>());
    osi<<x1<<" "<<x2<<endl;
    cout<<x1<<" "<<x2<<endl;
  }
  for(int i(0); i<50; ++i)
  {
    double x1(-0.5+0.1*NRand<double>()),x2(-0.5+0.1*NRand<double>());
    osi<<x1<<" "<<x2<<endl;
    cout<<x1<<" "<<x2<<endl;
  }
  return 0;
}
