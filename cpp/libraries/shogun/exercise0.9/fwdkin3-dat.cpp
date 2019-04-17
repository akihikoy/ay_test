#include "simpleode/lib/arm.h"
#include <cstdlib>
#include <cmath>
#include <fstream>

template<typename T>
inline T URand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}

using namespace std;

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static ofstream ofs_a("out-3d-angle.dat");
  static ofstream ofs_p("out-3d-pos.dat");
  static int count(0);
  static double target[3]={0,0,0};
  if(count%10==0)
  {
    target[0]= 2.0*(URand<double>()-0.5)*M_PI;
    target[1]= (URand<double>()-0.5)*M_PI;
    target[2]= 1.8*(URand<double>()-0.5)*M_PI;
  }
  ++count;
  const double kp(1.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(target[j]-robot.GetAngleHinge(j)));
  ofs_a<<robot.GetAngleHinge(0)<<" "<<robot.GetAngleHinge(1)<<" "<<robot.GetAngleHinge(2)<<endl;
  ofs_p<<robot.GetHandPos()[0]<<" "<<robot.GetHandPos()[1]<<" "<<robot.GetHandPos()[2]<<endl;
  if(count%100==0)  cerr<<"count: "<<count<<endl;
  if(count==2000)  exit(0);
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("textures",env);
  xode::ControlCallback= &CtrlCallback;
  xode::DrawCallback= &DrawCallback;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
