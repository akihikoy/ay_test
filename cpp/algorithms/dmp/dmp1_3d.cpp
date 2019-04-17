//-------------------------------------------------------------------------------------------
/*! \file    dmp1_3d.cpp
    \brief   dmp1_3d
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \date    Nov.26, 2013
*/
//-------------------------------------------------------------------------------------------
#include <lora/robot_model.h>
#include <lora/ode_ds.h>
// #include <lora/string.h>
// #include <lora/stl_math.h>
#include <lora/rand.h>
#include <fstream>
#include "dmp1.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
using namespace movement_primitives;
using namespace xode;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cerr<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

bool init_flag(true);

TWorld world;
dVector3 position, velocity, ang_vel, acceleration;
dQuaternion quaternion;
dMatrix3 rotation;

const int est_step(10);


double power[6]={0.0,0.0,0.0, 0.0,0.0,0.0};
double  ptime(0.0);

const dVector3 gravity={0.0,0.0,-9.8};
// const dVector3 gravity={0.0,0.0,0.0};

TDynamicMovementPrimitives<double> dmp;

ofstream ofs_dmp("res/dmp.dat");
ofstream ofs_can("res/can.dat");

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
  static int bidx= world.RootLinkBodyIndex("Robot");
  for(int i(0);i<3;++i) position[i]= w.Body(bidx).getPosition()[i];
  for(int i(0);i<3;++i) velocity[i]= w.Body(bidx).getLinearVel()[i];
  for(int i(0);i<3;++i) ang_vel[i]= w.Body(bidx).getAngularVel()[i];
  for(int i(0);i<4;++i) quaternion[i]= w.Body(bidx).getQuaternion()[i];
  for(int i(0);i<12;++i) rotation[i]= w.Body(bidx).getRotation()[i];

  // if(dmp.t_==0.0l)  dmp.z_= 0.0;
  // dmp.y_= position[0];
  dmp.Step(time_step, &position[0]);
  double kp(0.9),kd(0.1), u_max(1.1);
  double u= kp*(dmp.Y()-position[0])+kd*(dmp.dy_-velocity[0]);
  if(u>u_max)  u= u_max;
  else if(u<-u_max)  u= -u_max;
  world.Body(bidx).addForce(u,0.0,0.0);
  ofs_dmp<<world.Time()<<" "<<dmp.Y()<<endl;
  ofs_can<<world.Time()<<" "<<dmp.x_<<endl;

  if(ptime>0.0)
  {
    double w= (-(ptime-0.5)*(ptime-0.5)+0.25);
    world.Body(bidx).addForce(w*power[0],w*power[1],w*power[2]);
    world.Body(bidx).addTorque(w*power[3],w*power[4],w*power[5]);
    ptime-=time_step*5.0;
    if(ptime<=0.0)  for(int i(0);i<6;++i) power[i]= 0.0;
  }
}

void EndOfTimeStep(TWorld &w, const TReal &time_step)
{
  // static int bidx= world.RootLinkBodyIndex("Robot");
}

void EndOfDrawing(TWorld &w, const TReal &time_step)
{
#define _R(i,j) R[(i)*4+(j)]
  dMatrix3 R;
  // dRfromQ(R,OctBegin(est_quaternion));
  dRSetIdentity(R);
  dVector3 sides={0.4,0.3,0.2};
  dsSetColorAlpha(0.0, 0.0, 1.0, 0.3);
  // std::swap(_R(0,1),_R(1,0)); std::swap(_R(0,2),_R(2,0)); std::swap(_R(1,2),_R(2,1));
#undef _R
  dVector3 p={dmp.Y(),0.0,0.0};
  dsDrawBox(p, R, sides);
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
  const dReal f(1.0), fz(1.0), t(0.001), tz(0.01);
  switch(command)
  {
  case 'r': case 'R':
    world.Create();
    dmp.Init();
    init_flag= true;
    break;
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
  else        world.LoadFromFile("test_3d.var");
  world.Create();
  // print(world.TotalMass("Robot"));

  // DMP part
  {
    valarray<double> demo(100);
    double dt= 0.01;
    ofstream ofs_demo("res/demo.dat");
    for(int n(0); n<100; ++n)
    {
      double t= (double)n*dt;
      demo[n]= t+0.5*sin(10.809*t);
      ofs_demo<<t<<" "<<demo[n]<<endl;
    }
    dmp.SetAlphaZ(5.0);
    dmp.SetBetaZ(1.0);
    dmp.SetAlphaX(1.0);
    dmp.LearnFromDemo(demo, 1.0, 20);
  // print(dmp.tau_);
    // dt= 0.001;
    dmp.Init();
  // dmp.goal_= 0.6;
  // dmp.goal_= 2.0;
  // dmp.tau_*= 2.0;
  // dmp.tau_*= 0.5;
    // ofstream ofs_dmp("res/dmp.dat");
    // ofstream ofs_can("res/can.dat");
    // for(int n(0); n<3000; ++n)
    // {
      // double t= (double)n*dt;
      // dmp.Step(dt);
      // ofs_dmp<<t<<" "<<dmp.Y()<<endl;
  // // if(n>200&&n<600)  dmp.x_= dmp.x_-0.5*dt*dmp.dx_;
      // ofs_can<<t<<" "<<dmp.x_<<endl;
    // }
  }


  if(!world.ConsoleMode())
    dsSimulationLoop (argc,argv,400,400,&fn);
  else
    while(true)  world.Step();

  dCloseODE();

  // world.SaveToFile("result.dat");

  return 0;
}
//-------------------------------------------------------------------------------------------
