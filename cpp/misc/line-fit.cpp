//-------------------------------------------------------------------------------------------
/*! \file    line-fit.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.03, 2010
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
#include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
#include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TData {double x,z; TData(const double &x_,const double &z_):x(x_),z(z_){}};

void fit_line(const list<TData> &data, double &a, double &b)
{
  double X(0.0),Z(0.0),X2(0.0),ZX(0.0);
  for (list<TData>::const_iterator i(data.begin()),l(data.end());i!=l;++i)
  {
    X+=i->x;
    Z+=i->z;
    X2+=i->x*i->x;
    ZX+=i->z*i->x;
  }
  double N(data.size());
  b=(X*ZX-Z*X2)/(X*X-N*X2);
  a=ZX/X2-X/X2*b;
}

double f(const double &x) {return 3.0*x+2.0;}

int main(int argc, char**argv)
{
  const double noise(0.5);
  list<TData>  data;
  for (int i(0); i<1000; ++i)  {double x=Rand(0.0,5.0);data.push_back(TData(x,f(x)+Rand(-noise,noise)));}
  double a,b;
  fit_line(data,a,b);
  print(a);
  print(b);
  return 0;
}
//-------------------------------------------------------------------------------------------
