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
double func1(const double &x1,const double &x2)
{
  return exp(-2.0*x1*x1-4.0*x2*x2);
}
double func2(const double &x1,const double &x2)
{
  return -func1(x1,x2);
}
int main(int,char**)
{
  ofstream osi("d/input2.dat"),oso("d/output2.dat");
  const double noise(0.0);
  for(int i(0); i<400; ++i)
  {
    double x1(URand(-1.0,1.0)),x2(URand(-1.0,1.0));
    double y1(func1(x1,x2)+noise*NRand<double>());
    double y2(func2(x1,x2)+noise*NRand<double>());
    osi<<x1<<" "<<x2<<endl;
    oso<<y1<<" "<<y2<<endl;
    cout<<x1<<" "<<x2<<" "<<y1<<" "<<y2<<endl;
  }
  return 0;
}
