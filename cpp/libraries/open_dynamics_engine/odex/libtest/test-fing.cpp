#include "../lib/fing.h"
#include <iostream>
using namespace std;

void test_ctrl(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal target= 3.0*std::sin(2.0*env.Time());

  const double kp(3.0);
  robot.SetAngVelHinge(0,kp*(target-robot.GetAngleHinge(0)));
}

void test_draw(xode::TEnvironment &env, xode::TDynRobot &robot)
{
}

void test_keyevent(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("../textures",env);
  xode::ControlCallback= &test_ctrl;
  xode::DrawCallback= &test_draw;
  xode::KeyEventCallback= &test_keyevent;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
