//-------------------------------------------------------------------------------------------
/*! \file    manip01.cpp
    \brief   manip01
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \date    Apr.15, 2014
*/
//-------------------------------------------------------------------------------------------
#include <lora/robot_model.h>
#include <lora/ode_ds.h>
// #include <lora/string.h>
#include <lora/stl_math.h>
#include <lora/rand.h>
#include <fstream>
#include "manip01_ctrl.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
using namespace xode;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cerr<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

bool init_flag(true);

xode::TWorld world;
kinematics::TKinematicRobot kin_robot;

std::vector<TReal> targets(8,0.0);

dVector3 endeffector_target_pos={0,0,0};
dMatrix3 endeffector_target_rot={1,0,0,0, 0,1,0,0, 0,0,1,0};
#define CTRL_ROT

// dVector3 position, velocity, ang_vel, acceleration;
// dQuaternion quaternion;
// dMatrix3 rotation;

// const int est_step(10);


// double power[6]={0.0,0.0,0.0, 0.0,0.0,0.0};
// double  ptime(0.0);

const dVector3 gravity={0.0,0.0,-9.8};
// const dVector3 gravity={0.0,0.0,0.0};

void SimStart()
{
  world.Start();
}

void SimLoop(int pause)
{
  if(pause)
    world.StepDrawing();
  else
    while(!world.Step()) ;
}

void StartOfTimeStep(TWorld &w, const TReal &time_step)
{
  // static int bidx= world.RootLinkBodyIndex("Robot");
  // for(int i(0);i<3;++i) position[i]= w.Body(bidx).getPosition()[i];
  // for(int i(0);i<3;++i) velocity[i]= w.Body(bidx).getLinearVel()[i];
  // for(int i(0);i<3;++i) ang_vel[i]= w.Body(bidx).getAngularVel()[i];
  // for(int i(0);i<4;++i) quaternion[i]= w.Body(bidx).getQuaternion()[i];
  // for(int i(0);i<12;++i) rotation[i]= w.Body(bidx).getRotation()[i];

  static int dist_time(-1);
  ++dist_time;

  static int ridx= w.RobotIndex("Robot");
  if(ridx<0)  return;

  std::vector<TReal> angles(w.JointAngleNum(ridx));
  w.GetJointAngles(ridx,angles.begin(),angles.end());
  // PrintContainer(angles,"");

  std::vector<TReal> angvels(w.JointAngVelNum(ridx));
  w.GetJointAngVels(ridx,angvels.begin(),angvels.end());

  // std::vector<TReal> targets(w.JointAngleNum(ridx),0.0);
  // for(int j(0); j<w.JointAngleNum(ridx); ++j)
    // targets[j]= 1.0*real_sin(0.5l*w.Time());

#if 0
  {
    for(int j(0); j<w.JointAngleNum(ridx); ++j)
      targets[j]= 1.0*real_sin(0.5l*w.Time());
  }
#endif
  if(dist_time%20==0)
  {
    for(int j(0); j<kin_robot.MovableJointNum(); ++j)
      kin_robot.MovableJointAngle(j)= angles[j]; // 0.0;
    // endeffector_target_pos[0]= 0.2;
    // endeffector_target_pos[1]= 0.3*real_sin(1.0l*w.Time());
    // endeffector_target_pos[2]= 0.5;
    endeffector_target_pos[0]= 0.4;
    endeffector_target_pos[1]= 0.0;
    endeffector_target_pos[2]= 0.2;
    kin_robot.SetEndEffectorTargetPos ("L6", endeffector_target_pos);
  #ifdef CTRL_ROT
    // dRFromEulerAngles(endeffector_target_rot, Rand(-M_PI,M_PI),Rand(-M_PI,M_PI),Rand(-M_PI,M_PI));
    // dRFromEulerAngles(endeffector_target_rot, real_sin(0.6l*w.Time()), 0.0, 0.0);
    dRFromAxisAndAngle(endeffector_target_rot, 0,1,0, 0.5*M_PI);
    kin_robot.SetEndEffectorTargetRot ("L6", endeffector_target_rot);
  #endif
    bool is_solved= (kin_robot.ExecInverseKinematics(1.0e-6,100,0.3) < 1.0e-6);

    if(is_solved)
    {
      // targets[0]= kin_robot.GetJointAngle1("J1");
      // targets[1]= kin_robot.GetJointAngle1("J2");
      // targets[2]= kin_robot.GetJointAngle1("J3");
      // targets[3]= kin_robot.GetJointAngle1("J4");
      // targets[4]= kin_robot.GetJointAngle1("J5");
      // targets[5]= kin_robot.GetJointAngle1("J6");
      for(int j(0); j<kin_robot.MovableJointNum(); ++j)
        targets[j]= kin_robot.MovableJointAngle(j);
    }
    else  LDEBUG("IK failed @"<<w.Time());
  }
  targets[6]= targets[7]= 0.05;


  // PD controll
  std::vector<TReal> torques(w.JointTorqueInputNum(ridx));
  torques= 1.4l*(targets-angles) - 0.01l*angvels;
  w.AddToJointTorques(ridx,torques.begin(),torques.end());

  // if(ptime>0.0)
  // {
    // double w= (-(ptime-0.5)*(ptime-0.5)+0.25);
    // world.Body(bidx).addForce(w*power[0],w*power[1],w*power[2]);
    // world.Body(bidx).addTorque(w*power[3],w*power[4],w*power[5]);
    // ptime-=time_step*5.0;
    // if(ptime<=0.0)  for(int i(0);i<6;++i) power[i]= 0.0;
  // }
}

void EndOfTimeStep(TWorld &w, const TReal &time_step)
{
  // static int bidx= world.RootLinkBodyIndex("Robot");
}

void EndOfDrawing(TWorld &w, const TReal &time_step)
{
#define _R(i,j) R[(i)*4+(j)]
  // dMatrix3 R;
  // dRfromQ(R,OctBegin(est_quaternion));
  // dRSetIdentity(R);
  dVector3 sides={0.2,0.3,0.15};
  dsSetColorAlpha(0.0, 0.0, 1.0, 0.3);
  // std::swap(_R(0,1),_R(1,0)); std::swap(_R(0,2),_R(2,0)); std::swap(_R(1,2),_R(2,1));
#undef _R
  // dVector3 p={dmp.Y(),0.0,0.0};
  dsDrawBox(endeffector_target_pos, endeffector_target_rot, sides);
  // if(GenSize(est_dv)>0)
  // {
    // dVector3 pe;
    // for(int i(0);i<3;++i) pe[i]= position[i]+est_dv(i);
    // dsDrawLineD(position,pe);
  // }
}

void KeyEvent(int command)
{
  // static int bidx= world.RootLinkBodyIndex("Robot");
  // const dReal f(1.0), fz(1.0), t(0.001), tz(0.01);
  switch(command)
  {
  case 'r': case 'R':
    world.Create();
    for(int j(0); j<kin_robot.MovableJointNum(); ++j)
      kin_robot.MovableJointAngle(j)= 0.0;
    init_flag= true;
    break;
#if 0
  case ' ':
    ptime= 1.0;
    power[0]= f*Rand(-1.0,1.0);
    power[1]= f*Rand(-1.0,1.0);
    power[2]= fz*Rand(-1.0,1.0);
    break;
  case 'v': case 'V':
    ptime= 1.0;
    power[3]= t*Rand(-1.0,1.0);
    power[4]= t*Rand(-1.0,1.0);
    power[5]= tz*Rand(-1.0,1.0);
    break;
  case 'a': case 'A':
    // world.Body(bidx).addForce(f,0.0,0.0);
    ptime= 1.0;
    power[0]= f;
    break;
  case 'd': case 'D':
    // world.Body(bidx).addForce(-f,0.0,0.0);
    ptime= 1.0;
    power[0]= -f;
    break;
  case 'w': case 'W':
    // world.Body(bidx).addForce(0.0,f,0.0);
    ptime= 1.0;
    power[1]= f;
    break;
  case 's': case 'S':
    // world.Body(bidx).addForce(0.0,-f,0.0);
    ptime= 1.0;
    power[1]= -f;
    break;
#endif
  default:
    cout<<"key:"<<command<<endl;
  }
}

int main(int argc, char**argv)
{
  Srand();

  // setup pointers to drawstuff callback functions
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &SimStart;
  fn.step = &SimLoop;
  fn.command = &KeyEvent;
  fn.stop = 0;
  fn.path_to_textures = "textures";

  InitializeODE();

  world.SetCallbacks().StartOfTimeStep= &StartOfTimeStep;
  world.SetCallbacks().EndOfTimeStep= &EndOfTimeStep;
  world.SetCallbacks().EndOfDrawing= &EndOfDrawing;
  if(argc>1)  world.LoadFromFile(argv[1]);
  else        world.LoadFromFile("env1.var");
  world.Create();
  GenerateKinematicRobotModel(kin_robot, world.RobotParams(world.RobotIndex("Robot")));

  kin_robot.AddEndEffectorPos("L6");
#ifdef CTRL_ROT
  kin_robot.AddEndEffectorRot("L6");
#endif

  kin_robot.AddMovableJoint("J1");
  kin_robot.AddMovableJoint("J2");
  kin_robot.AddMovableJoint("J3");
  kin_robot.AddMovableJoint("J4");
  kin_robot.AddMovableJoint("J5");
  kin_robot.AddMovableJoint("J6");

  // print(world.TotalMass("Robot"));

  if(!world.ConsoleMode())
    dsSimulationLoop (argc,argv,400,400,&fn);
  else
    while(true)  world.Step();

  dCloseODE();

  // world.SaveToFile("result.dat");

  return 0;
}
//-------------------------------------------------------------------------------------------
