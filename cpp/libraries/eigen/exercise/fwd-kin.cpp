#include "../lib/arm.h"
#include <Eigen/Dense>

void ForwardKin(const Eigen::Vector3d &q, Eigen::Affine3d &Te, Eigen::Matrix3d &J)
{
  using namespace Eigen;
  const double l1(0.1),l2(0.1),l3(0.6),l4(0.6);

  // Matrix4d T12,T23,T34,T4e;
  // T4e<< Matrix3d::Identity(), Vector3d(0.0,0.0,l4),
  //       RowVector3d::Zero(), 1.0;
  // T34<< AngleAxisd(q(2),Vector3d::UnitY()).toRotationMatrix(), Vector3d(0.0,0.0,l3),
  //       RowVector3d::Zero(), 1.0;
  // T23<< AngleAxisd(q(1),Vector3d::UnitY()).toRotationMatrix(), Vector3d(0.0,0.0,l2),
  //       RowVector3d::Zero(), 1.0;
  // T12<< AngleAxisd(q(0),Vector3d::UnitZ()).toRotationMatrix(), Vector3d(0.0,0.0,l1),
  //       RowVector3d::Zero(), 1.0;
  // Matrix4d Te= T12*T23*T34*T4e;

  Affine3d T12,T23,T34,T4e;
  T4e= Translation3d(0.0,0.0,l4);
  T34= Translation3d(0.0,0.0,l3) * AngleAxisd(q(2),Vector3d::UnitY());
  T23= Translation3d(0.0,0.0,l2) * AngleAxisd(q(1),Vector3d::UnitY());
  T12= Translation3d(0.0,0.0,l1) * AngleAxisd(q(0),Vector3d::UnitZ());
  // Matrix4d T= (T12*T23*T34*T4e).matrix();
  Affine3d T3=T12*T23, T4=T3*T34;
  Te=T4*T4e;

  Vector3d pos= Te.translation();
  J<< (T12.linear()*Vector3d::UnitZ()).cross(pos-T12.translation()),
      (T3.linear()*Vector3d::UnitY()).cross(pos-T3.translation()),
      (T4.linear()*Vector3d::UnitY()).cross(pos-T4.translation());
}

#ifndef NOT_MAIN

using namespace Eigen;

#define print(var)  \
  std::cout<<#var" =\n"<<(var)<<std::endl

void CtrlCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  dReal target[3]={0.5,0.8,0.8};
  target[0]= std::sin(env.Time());
  target[1]= 0.5*(std::sin(0.2+2*env.Time())+1.0);
  target[2]= 1.57*0.5;
  const double kp(5.0);
  for(int j(0);j<3;++j)
    robot.SetAngVelHinge(j,kp*(target[j]-robot.GetAngleHinge(j)));
}

void DrawCallback(xode::TEnvironment &env, xode::TDynRobot &robot)
{
  Vector3d q(robot.GetAngleHinge(0),robot.GetAngleHinge(1),robot.GetAngleHinge(2));
  Vector3d dq(robot.GetAngVelHinge(0),robot.GetAngVelHinge(1),robot.GetAngVelHinge(2));

  Affine3d Te;
  Matrix3d J;
  ForwardKin(q, Te, J);

  print(Te.matrix());
  print(J);

  Vector3d pos= Te.translation();
  Vector3d v= J*dq, p2= pos+0.5*v;

  dReal sides[3]={0.3,0.3,0.3};
  dsSetColorAlpha (0.0, 0.0, 1.0, 0.5);
  Matrix4d Tt=Te.matrix().transpose();
  dsDrawBox (pos.data(), Tt.data(), sides);
  dsDrawLine (pos.data(), p2.data());
}

int main (int argc, char **argv)
{
  xode::TEnvironment env;
  dsFunctions fn= xode::SimInit("../textures",env);
  xode::ControlCallback= &CtrlCallback;
  xode::DrawCallback= &DrawCallback;

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
#endif // NOT_MAIN
