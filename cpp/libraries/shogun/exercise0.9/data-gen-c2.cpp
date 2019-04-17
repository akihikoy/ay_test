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
int func(const double &x1,const double &x2)
{
  if (Sq(x1-0.5)+0.2*(x1-0.5)*(x2-0.4)+2*Sq(x2-0.4) < 1) return -1;
  if (2*Sq(x1+0.5)+0.2*(x1+0.5)*(x2+0.2)+Sq(x2+0.2) < 1) return -1;
  return +1;
}
int main(int,char**)
{
  ofstream osi("d/input4.dat");
  for(int i(0); i<200; ++i)
  {
    double x1(0.1+1.0*NRand<double>()),x2(0.1+1.0*NRand<double>());
    double y(func(x1,x2));
    if(y==-1)  osi<<x1<<" "<<x2<<endl;
    cout<<x1<<" "<<x2<<" "<<y<<endl;
  }
  return 0;
}
