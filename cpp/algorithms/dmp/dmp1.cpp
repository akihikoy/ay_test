//-------------------------------------------------------------------------------------------
/*! \file    dmp1.cpp
    \brief   DMP test code
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.25, 2013
*/
//-------------------------------------------------------------------------------------------
#include "dmp1.h"
#include <fstream>
//-------------------------------------------------------------------------------------------
namespace movement_primitives
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace movement_primitives;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  valarray<double> demo(100);
  double dt= 0.01;
  ofstream ofs_demo("res/demo.dat");
  for(int n(0); n<100; ++n)
  {
    double t= (double)n*dt;
    demo[n]= t+0.5*sin(12.0*t);
    ofs_demo<<t<<" "<<demo[n]<<endl;
  }
  TDynamicMovementPrimitives<double> dmp;
  dmp.SetAlphaZ(1.0);
  dmp.SetBetaZ(2.0);
  dmp.SetAlphaX(1.0);
  dmp.LearnFromDemo(demo, 1.0, 20);
// print(dmp.tau_);
  dt= 0.001;
  dmp.Init();
// dmp.goal_= 0.6;
// dmp.goal_= 2.0;
dmp.tau_*= 2.0;
// dmp.tau_*= 0.5;
  ofstream ofs_dmp("res/dmp.dat");
  ofstream ofs_can("res/can.dat");
  for(int n(0); n<3000; ++n)
  {
    double t= (double)n*dt;
    dmp.Step(dt);
    ofs_dmp<<t<<" "<<dmp.Y()<<endl;
// if(n>200&&n<600)  dmp.x_= dmp.x_-0.5*dt*dmp.dx_;
    ofs_can<<t<<" "<<dmp.x_<<endl;
  }
  ofstream ofs_kernel("res/kernel.dat");
ofstream ofs_f("res/f.dat");
  for(double x(0.001);x<1.0;x+=0.001)
  {
    std::valarray<double> phi;
    dmp.get_phi(x, phi);
    ofs_kernel<<x<<" "<<Eigen::Map<Eigen::VectorXd>(&phi[0],phi.size()).transpose()/x<<endl;
    // ofs_kernel<<x<<" "<<Eigen::Map<Eigen::VectorXd>(&phi[0],phi.size()).transpose()<<endl;
double f= std::inner_product(&phi[0],&phi[0]+phi.size(),&dmp.w_[0],0.0l)*(dmp.goal_-dmp.start_);
ofs_f<<x<<" "<<f<<endl;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
