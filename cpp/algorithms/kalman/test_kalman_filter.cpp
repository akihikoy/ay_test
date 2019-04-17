//-------------------------------------------------------------------------------------------
/*! \file    test_kalman_filter.cpp
    \brief   test kalman filter for linear model
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.21, 2013

    test code by:
      ./a.out > ! res.dat && qplot res.dat u 2:3 w l res.dat u 6:7 w lp && qplot res.dat u 1:4 res.dat u 1:8 w l res.dat u 1:5 res.dat u 1:9 w l
*/
//-------------------------------------------------------------------------------------------
#include "kalman_filter.h"
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
#include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/type_gen_oct.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
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
#define print(var) std::cerr<<#var"= "<<std::endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  Srand();
  TKalmanFilter kf;
  TKalmanFilter::TVector x(2,0.0), v(2,0.0), v_obs(2,0.0), noise(2,0.0), u, x_est(2,0.0);
  double dt(0.01), dx(0.1), dv(0.2), max_t(50.0);
  int obs_cycle(10), oid(0);
  double obs_dt=dt*(double)obs_cycle;

  kf.SetA(OctGen2<TKalmanFilter::TMatrix>(4,4, 1.0,0.0,obs_dt,0.0, 0.0,1.0,0.0,obs_dt, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0));
  kf.SetC(OctGen2<TKalmanFilter::TMatrix>(2,4, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0));
  kf.SetR(GetEye(4)*dx*dx);
  kf.SetQ(GetEye(2)*dv*dv*9.0);

  print(kf.GetA());
  print(kf.GetC());
  print(kf.GetR());
  print(kf.GetQ());

  kf.Initialize(OctGen1<TKalmanFilter::TVector>(4, 0.0,0.0,0.0,0.0), GetEye(4));
  for(double t(0.0); t<max_t; t+=dt)
  {
    #if 0
    if(t<2.0)  {v(0)=0.0; v(1)= 0.0;}
    else if(t<3.0)  {v(0)=1.0; v(1)= 0.0;}
    else if(t<4.0)  {v(0)=0.0; v(1)= 1.0;}
    else if(t<5.0)  {v(0)=1.0; v(1)= 1.0;}
    else if(t<6.0)  {v(0)=1.0; v(1)= -1.0;}
    else  {v(0)=0.0; v(1)= 0.0;}
    #endif
    #if 1
    v(0)=real_cos(4.0*t)+real_sin(6.0*t)+0.05; v(1)=real_sin(8.0*t)+1.0;
    #endif
    x= x + v*dt;
    if(oid==0)
    {
      noise= OctGen1<TKalmanFilter::TVector>(2, (double)Rand(-dv,dv), (double)Rand(-dv,dv));
      // if(t>7.0)  {noise= noise*5.0;}
      v_obs= v+noise;
      kf.Update(u, v_obs);
      x_est= x_est + v_obs*obs_dt;
    }
    ++oid; if(oid==obs_cycle) oid=0;
    cout<<t<<" "<<(x).transpose()<<" "<<(v_obs).transpose()<<" "<<(kf.GetMu()).transpose()<<endl;
    cerr<<(kf.GetSigma())<<endl;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
