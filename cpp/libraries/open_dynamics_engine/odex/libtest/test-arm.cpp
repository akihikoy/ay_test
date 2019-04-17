#include "../lib/arm.h"
#include <iostream>
using namespace std;

void test_ctrl(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal target[3]={0.5,0.8,0.8};

  // robot.Body().back().addForce (0.1*(std::sin(env.Time())+1.0), 0.1*(std::sin(env.Time()*1.7)+1.0) ,0.1*(std::sin(env.Time()*0.7)+1.0));

  target[0]= std::sin(env.Time());
  target[1]= 0.5*(std::sin(0.2+2.0*env.Time())+1.0);
  target[2]= 1.57*0.5;

  const double kp(5.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(target[j]-robot.GetAngleHinge(j)));

  // const double kp(0.2),kd(0.02);
  // for(int j(0);j<3;++j)
  //   robot.AddTorqueHinge(j,kp*(target[j]-robot.GetAngleHinge(j))-kd*robot.GetAngVelHinge(j));

  cerr<<robot.GetAngleHinge(0)<<" "<<robot.GetAngleHinge(1)<<" "<<robot.GetAngleHinge(2)<<endl;
}

static dVector3 KeyboardShperePos={0,0,1};

void test_draw(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal sides[3]={0.3,0.3,0.3};
  dsSetColorAlpha (0.0, 1.0, 1.0, 0.5);
  dsDrawBox (robot.GetHandPos(), robot.GetHandRot(), sides);
  dsSetColorAlpha (1.0, 0.0, 1.0, 0.5);
  // dsDrawSphere (KeyboardShperePos, robot.GetHandRot(), 0.2);
}

void test_keyevent(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
  dReal d(0.05);
  switch(command)
  {
  case 'd': KeyboardShperePos[0]+=d; break;
  case 'a': KeyboardShperePos[0]-=d; break;
  case 'w': KeyboardShperePos[2]+=d; break;
  case 's': KeyboardShperePos[2]-=d; break;
  case 'e': KeyboardShperePos[1]+=d; break;
  case 'q': KeyboardShperePos[1]-=d; break;
  }
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("../textures",env);
  // dsFunctions fn= xode::SimInit("/home/akihiko/proj/skyai/benchmarks/humanoid01/m/textures_w",env);
  xode::ControlCallback= &test_ctrl;
  xode::DrawCallback= &test_draw;
  xode::KeyEventCallback= &test_keyevent;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
