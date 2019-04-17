#define NOT_MAIN
#include "inv-kin.cpp"

using namespace Eigen;

#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl

Vector3d Target(0.7,0.7,0.6);

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  static int count(0);
  static Vector3d q(0,0,0);
  if(count%1==0)
  {
    q= Vector3d(robot.GetAngleHinge(0),robot.GetAngleHinge(1),robot.GetAngleHinge(2));
    IKParam ikparam;
    ikparam.StepSize=0.3;
    ikparam.Tolerance=1.e-4;
    InverseKin(Target,q,ikparam);
    print(q.transpose());
  }
  ++count;

  const double kp(10.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(q[j]-robot.GetAngleHinge(j)));
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  // draw IK result:
  dReal sides[3]={0.3,0.3,0.3};
  dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  Matrix4d I=Matrix4d::Identity();
  dsDrawBox (Target.data(), I.data(), sides);
}

void KeyEventCallback(xode::TEnvironment &env, xode::TDynRobot &robot, int command)
{
  dReal d(0.05);
  switch(command)
  {
  case 'd': Target[0]+=d; break;
  case 'a': Target[0]-=d; break;
  case 'w': Target[2]+=d; break;
  case 's': Target[2]-=d; break;
  case 'e': Target[1]+=d; break;
  case 'q': Target[1]-=d; break;
  }
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("../textures",env);
  xode::ControlCallback= &CtrlCallback;
  xode::DrawCallback= &DrawCallback;
  xode::KeyEventCallback= &KeyEventCallback;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
