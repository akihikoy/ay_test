#include "simpleode/lib/fing.h"
#include <iostream>
#include <fstream>
using namespace std;

double Params[2]={3.0,2.0};

void set_params()
{
  Params[0]= 2.0*dRandReal()+1.0;
  Params[1]= 2.0*dRandReal()+1.0;
}

void test_ctrl(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static ofstream ofs_params("out-params.dat");
  static ofstream ofs_pos("out-pos.dat");

  if(env.Time()>3.0 && robot.GetBodyContact(3))
  {
    ofs_params<<Params[0]<<" "<<Params[1]<<endl;
    ofs_pos<<robot.Body(3).getPosition()[0]<<endl;
    env.Reset();
    set_params();
    return;
  }

  dReal target= Params[0]*std::sin(Params[1]*env.Time());

  const double kp(3.0);
  robot.SetAngVelHinge(0,kp*(target-robot.GetAngleHinge(0)));
}

void test_draw(xode::TEnvironment &env, xode::TDynRobot &robot)
{
}

void test_keyevent(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
  if(command=='r')  env.Reset();
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("textures",env);
  xode::ControlCallback= &test_ctrl;
  xode::DrawCallback= &test_draw;
  xode::KeyEventCallback= &test_keyevent;

  cerr<<"Push \'R\' to reset the simpulator"<<endl;

  set_params();
  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
