//-------------------------------------------------------------------------------------------
/*! \file    test_3d.cpp
    \brief   test_3d
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \date    Oct.8, 2013
*/
//-------------------------------------------------------------------------------------------
#include "q_ekf.h"
#include <lora/robot_model.h>
#include <lora/ode_ds.h>
#include <lora/string.h>
#include <lora/stl_math.h>
#include <lora/rand.h>
#include <lora/octave.h>
#include <lora/type_gen_oct.h>
#include <lora/small_classes.h>  // TMovingAverageFilter
#include <fstream>
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

typedef ColumnVector TVector;
typedef Matrix TMatrix;

bool init_flag(true);

TWorld world;
dVector3 position, position_old, velocity, ang_vel, velocity_old={0,0,0}, acceleration;
dQuaternion quaternion;
dMatrix3 rotation;

const int est_step(10);

TVector est_position, est_dv;
TVector est_quaternion;

TVector power(6,0.0);
double  ptime(0.0);

const dVector3 gravity={0.0,0.0,-9.8};
// const dVector3 gravity={0.0,0.0,0.0};

TExtendedKalmanFilter ekf;

void SetupEKF()
{
  // TReal dt(0.0001), dx(0.1), dv(0.2);
  TReal dt(0.0001*TReal(est_step)), dx(0.00), dv(0.00);
  TKalmanFilter::TVector g(3,0.0), noise_mean(3,0.0);
  g(0)= gravity[0]; g(1)= gravity[1]; g(2)= gravity[2];
  // TKalmanFilter::TMatrix R(GetEye(16)*dx*dx), Q(GetEye(6)*dv*dv);
  TKalmanFilter::TMatrix R(GetEye(16)*dx*dx), Q(GetEye(6)*dv*dv);
  R(10,10)= R(11,11)= R(12,12)= 0.04;
  R(13,13)= R(14,14)= R(15,15)= 0.001;

  using namespace q_ekf;
  ekf.SetStateTransModel(boost::bind(&func_state_trans,_1,_2,dt));
  ekf.SetObservationModel(boost::bind(&func_observation,_1,dt,g,noise_mean));
  ekf.SetStateConstraint(boost::bind(&func_constrain_state,_1));
  ekf.SetG(boost::bind(&func_G,_1,_2,dt));
  ekf.SetH(boost::bind(&func_H,_1,dt,g));
  ekf.SetR(R);
  ekf.SetQ(Q);

  TKalmanFilter::TVector init_mu(16,0.0), init_q(4,0.0);
  init_q(0)= 1.0;
  q_ekf::detail::ASIGN_Q(init_mu,init_q);
  // ekf.Initialize(init_mu, GetEye(16));
  ekf.Initialize(init_mu, GetEye(16)*0.0);
}

void Estimate(const TVector &sensor_ang_vel, const TVector &sensor_acc, TVector &est_position, TVector &est_quaternion, const double &dt)
{
LMESSAGE("--------------------");
  GenResize(est_position,3);
  GenResize(est_quaternion,4);
  // for(int i(0);i<3;++i) est_position(i)= position[i];
  // for(int i(0);i<4;++i) est_quaternion(i)= quaternion[i];
  // for(int i(1);i<4;++i) est_quaternion(i)*= -1.0;
// LDBGVAR(sensor_acc.transpose());
// LDBGVAR(sensor_ang_vel.transpose());

  // write kalman filter code here:
  // static TVector p(3,0.0), v(3,0.0), a_g(3,0.0);
  // a_g(2)=gravity[2];
  // v+= dt*(sensor_acc-a_g);
  // p+= dt*v;
  // est_position= p;
  // est_quaternion(0)= 1.0;
  static TKalmanFilter::TVector u, v_obs(6,0.0);
  v_obs.insert(sensor_ang_vel,0);
  v_obs.insert(sensor_acc,3);
  // v_obs(0)= 22.0;
  // v_obs(1)= 22.0;
  // v_obs(2)= 22.0;
  // v_obs(3)= 22.0;
  // v_obs(4)= 22.0;
  // v_obs(5)= 22.0;
static ofstream ofs("debug.dat"); ofs<<v_obs.transpose()<<endl;
LDBGVAR(v_obs.transpose());
  ekf.Update(u, v_obs);
  est_position= q_ekf::detail::EXT_P(ekf.GetMu());
  est_quaternion= q_ekf::detail::EXT_Q(ekf.GetMu());
  est_dv= q_ekf::detail::EXT_V(ekf.GetMu());
LDBGVAR(ekf.GetMu().transpose());
LMESSAGE("--------------------");
}

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
  if(init_flag)  {for(int i(0);i<3;++i) velocity[i]= 0.0;}
  static int bidx= world.RootLinkBodyIndex("Robot");
  for(int i(0);i<3;++i) position_old[i]= w.Body(bidx).getPosition()[i];
  for(int i(0);i<3;++i) velocity_old[i]= velocity[i];

  if(ptime>0.0)
  {
    double w= (-(ptime-0.5)*(ptime-0.5)+0.25);
    world.Body(bidx).addForce(w*power(0),w*power(1),w*power(2));
    world.Body(bidx).addTorque(w*power(3),w*power(4),w*power(5));
    ptime-=time_step*5.0;
    if(ptime<=0.0)  power= TVector(6,0.0);
  }
}

void EndOfTimeStep(TWorld &w, const TReal &time_step)
{
  static int bidx= world.RootLinkBodyIndex("Robot");
  for(int i(0);i<3;++i) position[i]= w.Body(bidx).getPosition()[i];
  for(int i(0);i<3;++i) velocity[i]= (position[i]-position_old[i])/time_step; // w.Body(bidx).getLinearVel()[i];
  for(int i(0);i<3;++i) ang_vel[i]= w.Body(bidx).getAngularVel()[i];
  for(int i(0);i<3;++i) acceleration[i]= (velocity[i]-velocity_old[i])/time_step;
  for(int i(0);i<4;++i) quaternion[i]= w.Body(bidx).getQuaternion()[i];
  for(int i(0);i<12;++i) rotation[i]= w.Body(bidx).getRotation()[i];

  static TMovingAverageFilter<TVector> acceleration_maf;
  if(init_flag)  {acceleration_maf.Initialize(est_step*5,TVector(3,0.0));  init_flag=false;}
  TVector tmp_acc(3,0.0); for(int i(0);i<3;++i) tmp_acc(i)= acceleration[i];
  acceleration_maf.Step(tmp_acc);
LDBGVAR(acceleration_maf().transpose());

  static int i_est(1);
  if(i_est%est_step==0)
  {
    double dw(0.00), da(0.00);
    TVector sensor_ang_vel(3);
    TVector sensor_acc(3);
    for(int i(0);i<3;++i) sensor_ang_vel(i)= ang_vel[i];
    // for(int i(0);i<3;++i) sensor_acc(i)= acceleration[i]+gravity[i];
    for(int i(0);i<3;++i) sensor_acc(i)= acceleration_maf()(i)+gravity[i];
    TMatrix rot(3,3);
    #define _R(i,j) rotation[(i)*4+(j)]
    for(int r(0);r<3;++r) for(int c(0);c<3;++c) rot(r,c)= _R(r,c);
    #undef _R
    sensor_ang_vel= rot.transpose()*sensor_ang_vel;
    sensor_acc= rot.transpose()*sensor_acc;
    for(int i(0);i<3;++i) sensor_ang_vel(i)+= dw*Rand(-1.0,1.0);
    for(int i(0);i<3;++i) sensor_acc(i)+= da*Rand(-1.0,1.0);

    Estimate(sensor_ang_vel, sensor_acc, est_position, est_quaternion, time_step*est_step);
  }
  ++i_est;
}

void EndOfDrawing(TWorld &w, const TReal &time_step)
{
#define _R(i,j) R[(i)*4+(j)]
  dMatrix3 R;
  dRfromQ(R,OctBegin(est_quaternion));
  dVector3 sides={0.4,0.3,0.2};
  dsSetColorAlpha(0.0, 0.0, 1.0, 0.3);
  // std::swap(_R(0,1),_R(1,0)); std::swap(_R(0,2),_R(2,0)); std::swap(_R(1,2),_R(2,1));
#undef _R
  if(GenSize(est_position)>0)
    dsDrawBox(OctBegin(est_position), R, sides);
  if(GenSize(est_dv)>0)
  {
    dVector3 pe;
    for(int i(0);i<3;++i) pe[i]= position[i]+est_dv(i);
    dsDrawLineD(position,pe);
  }
}

void KeyEvent(int command)
{
  // static int bidx= world.RootLinkBodyIndex("Robot");
  const dReal f(0.1), fz(1.0), t(0.001), tz(0.01);
  switch(command)
  {
  case 'r': case 'R':
    world.Create();
    SetupEKF();
    init_flag= true;
    break;
  case ' ':
    ptime= 1.0;
    power(0)= f*Rand(-1.0,1.0);
    power(1)= f*Rand(-1.0,1.0);
    power(2)= fz*Rand(-1.0,1.0);
    break;
  case 'v': case 'V':
    ptime= 1.0;
    power(3)= t*Rand(-1.0,1.0);
    power(4)= t*Rand(-1.0,1.0);
    power(5)= tz*Rand(-1.0,1.0);
    break;
  case 'a': case 'A':
    // world.Body(bidx).addForce(f,0.0,0.0);
    ptime= 1.0;
    power(0)= f;
    break;
  case 'd': case 'D':
    // world.Body(bidx).addForce(-f,0.0,0.0);
    ptime= 1.0;
    power(0)= -f;
    break;
  case 'w': case 'W':
    // world.Body(bidx).addForce(0.0,f,0.0);
    ptime= 1.0;
    power(1)= f;
    break;
  case 's': case 'S':
    // world.Body(bidx).addForce(0.0,-f,0.0);
    ptime= 1.0;
    power(1)= -f;
    break;
  default:
    cout<<"key:"<<command<<endl;
  }
}

int main(int argc, char**argv)
{
  Srand();
  SetupEKF();

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

  if(!world.ConsoleMode())
    dsSimulationLoop (argc,argv,400,400,&fn);
  else
    while(true)  world.Step();

  dCloseODE();

  // world.SaveToFile("result.dat");

  return 0;
}
//-------------------------------------------------------------------------------------------
