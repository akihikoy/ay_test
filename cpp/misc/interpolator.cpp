//-------------------------------------------------------------------------------------------
/*! \file    interpolator.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Sep.16, 2012
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
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
#include <iostream>
// #include <iomanip>
// #include <string>
#include <cmath>
// #include <list>
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

template<typename T>
inline T Square(const T &x)  {return x*x;}

void interpolate(const double &q0, const double &qt, const double &v, const double &a)
{
  double d= qt-q0;
  double s= (qt>=q0?1.0:-1.0);
  double t1= v/a;
  double tf= t1 + d/(s*v);
  double t2= tf - t1;
  double x1= 0.5*s*v*v/a;
  double x2= d - x1;

  double x;
  double time_step= 0.05;
  for(double t(0.0);t<tf;t+=time_step)
  {
    if(t<t1)  x= 0.5*s*a*Square(t);
    else if(t<t2)  x= x1 + s*v*(t-t1);
    else x= x2 + s*v*(t-t2) - 0.5*s*a*Square(t-t2);
    x+= q0;
    cout<<t<<" "<<x<<endl;
  }
}

class TInterpolator
{
public:
  void Init(const double &q0, const double &qt, const double &v, const double &a)
    {
      t_= 0.0;
      q0_= q0;
      qt_= qt;
      v_= v;
      a_= a;
      d_= qt_-q0_;
      s_= (qt_>=q0_?1.0:-1.0);
      if(s_*d_ > v_*v_/a_)
      {
cerr<<"xxxx"<<endl;
        t1_= v_/a_;
        tf_= t1_ + d_/(s_*v_);
        t2_= tf_ - t1_;
        x1_= 0.5*s_*v_*v_/a_;
        x2_= d_ - x1_;
        v2_= v_;
      }
      else
      {
        tf_= 2.0*std::sqrt(s_*d_/a_);
        t1_= 0.5*tf_;
        t2_= t1_;
        x1_= 0.5*d_;
        x2_= x1_;
        v2_= a_*t1_;
cerr<<"q0_:"<<q0_<<endl;
cerr<<"qt_:"<<qt_<<endl;
cerr<<"t1_:"<<t1_<<endl;
cerr<<"t2_:"<<t2_<<endl;
cerr<<"tf_:"<<tf_<<endl;
cerr<<"x1_:"<<x1_<<endl;
cerr<<"x2_:"<<x2_<<endl;
cerr<<"d_:"<<d_<<endl;
cerr<<"a_*tf_*tf_*0.25: "<<a_*tf_*tf_*0.25<<endl;
t_= t2_;
cerr<<"x1_ + s_*v_*(t_-t1_): "<<x1_ + s_*v_*(t_-t1_)<<endl;
t_= tf_;
cerr<<"x2_ + s_*v_*(t_-t2_) - 0.5*s_*a_*Square(t_-t2_): "<<x2_ + s_*v_*(t_-t2_) - 0.5*s_*a_*Square(t_-t2_)<<endl;
t_= 0;
      }
    }
  double Step(const double &time_step)
    {
      double x;
      t_+= time_step;
      if(t_<t1_)  x= 0.5*s_*a_*Square(t_);
      else if(t_<t2_)  x= x1_ + s_*v_*(t_-t1_);
      else if(t_<tf_) x= x2_ + s_*v2_*(t_-t2_) - 0.5*s_*a_*Square(t_-t2_);
      else x= d_;
      x+= q0_;
      return x;
    }
  const double& Tf() const {return tf_;}
private:
  double t_;
  double q0_;
  double qt_;
  double v_;
  double a_;
  double d_;
  double s_;
  double t1_;
  double tf_;
  double t2_;
  double x1_;
  double x2_;
  double v2_;
};


int main(int argc, char**argv)
{
  const double a(0.005),v(0.1);
  interpolate(0.0, 2.0, v,a);
  interpolate(2.0, 1.0, v,a);
  interpolate(1.0, -1.0, v,a);
  interpolate(-1.0, 2.0, v,a);

  TInterpolator i;
  i.Init(0.0, 2.0, v,a);
  for(double t(0.0); t<i.Tf(); t+= 0.05)
    cout<<t<<" "<<i.Step(0.05)<<endl;
  i.Init(2.0, 1.0, v,a);
  for(double t(0.0); t<i.Tf(); t+= 0.05)
    cout<<t<<" "<<i.Step(0.05)<<endl;
  i.Init(1.0, -1.0, v,a);
  for(double t(0.0); t<i.Tf(); t+= 0.05)
    cout<<t<<" "<<i.Step(0.05)<<endl;
  i.Init(-1.0, 2.0, v,a);
  for(double t(0.0); t<i.Tf(); t+= 0.05)
    cout<<t<<" "<<i.Step(0.05)<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
