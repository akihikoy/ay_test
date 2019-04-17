#ifdef NOT_MAIN
# include "fwd-kin.cpp"  // for ForwardKin
# include "inv.cpp"  // for PseudoInverse
#else
# define NOT_MAIN
# include "fwd-kin.cpp"  // for ForwardKin
# include "inv.cpp"  // for PseudoInverse
# undef NOT_MAIN
#endif

// matlab-like mod function that returns always positive
template<typename T>
inline T Mod(const T &x, const T &y)
{
  if(y==0)  return x;
  return x-y*std::floor(x/y);
}
// convert radian to [-pi,pi)
double RadToNPiPPi(const double &x)
{
  return Mod(x+M_PI,M_PI*2.0)-M_PI;
}

struct IKParam
{
  double StepSize;
  double Tolerance;
  int    MaxIteration;
  IKParam() :
    StepSize(0.3),
    Tolerance(1.e-6),
    MaxIteration(100)
    {}
};
void InverseKin(const Eigen::Vector3d &target, Eigen::Vector3d &q, const IKParam &param=IKParam())
{
  using namespace Eigen;
  Affine3d Te;
  Matrix3d J;
  Vector3d err;
  for(int count(0);count<param.MaxIteration;++count)
  {
    ForwardKin(q, Te, J);
    err= target - Te.translation();
    if(err.norm() < param.Tolerance)  break;
    q+= param.StepSize*PseudoInverse(J)*err;
    q= q.unaryExpr(&RadToNPiPPi);
    /*debug*/std::cerr<<"+"<<count<<" "<<Te.translation().transpose()<<" ("<<err.norm()<<")"<<std::endl;
  }
}

#ifndef NOT_MAIN

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
    Target(1)=0.5*sin(2.0*env.Time());
    Target(2)=0.6+0.3*cos(2.0*env.Time());
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
