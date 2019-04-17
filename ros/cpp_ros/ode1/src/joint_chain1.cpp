//-------------------------------------------------------------------------------------------
/*! \file    joint_chain1.cpp
    \brief   ODE N-joint chain simulation 1.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.05, 2015
*/
//-------------------------------------------------------------------------------------------
#include "ode1/joint_chain1.h"
#include <iostream>
#include <cassert>
//-------------------------------------------------------------------------------------------
namespace trick
{
using namespace std;
// using namespace boost;


//-------------------------------------------------------------------------------------------
namespace ode_x
{

int MaxContacts(4);  // maximum number of contact points per body
double VizJointLen(0.1);
double VizJointRad(0.02);
int    JointNum(6);
double TotalLen(1.0);
double LinkRad(0.03);
std::vector<double> TargetAngles(JointNum,0.0);
double TimeStep(0.04);
double Gravity(-0.5);
bool Running(true);
void (*SensingCallback)(const TSensors1 &sensors)= NULL;
void (*DrawCallback)(void)= NULL;
void (*StepCallback)(const double &time, const double &time_step)= NULL;
//-------------------------------------------------------------------------------------------

static TEnvironment *Env(NULL);
static void NearCallback(void*,dGeomID,dGeomID);
//-------------------------------------------------------------------------------------------


dReal IndexedColors[][4]= {
    /*0*/ {1.0, 0.0, 0.0, 0.8},
          {0.0, 1.0, 0.0, 0.8},
          {0.0, 0.0, 1.0, 0.8},
    /*3*/ {1.0, 1.0, 0.0, 0.8},
          {1.0, 0.0, 1.0, 0.8},
          {0.0, 1.0, 1.0, 0.8},
    /*6*/ {1.0, 0.0, 0.0, 0.3},
          {0.0, 1.0, 0.0, 0.3},
          {0.0, 0.0, 1.0, 0.3},
    /*9*/ {1.0, 1.0, 0.0, 0.3},
          {1.0, 0.0, 1.0, 0.3},
          {0.0, 1.0, 1.0, 0.3},
    /*12*/{1.0, 1.0, 1.0, 0.8}};
inline void SetColor(int i)
{
  dReal *col= IndexedColors[i%13];
  dsSetColorAlpha(col[0],col[1],col[2],col[3]);
}
//-------------------------------------------------------------------------------------------

/* ODE::dBody pose to pose.
    x: [0-2]: position x,y,z, [3-6]: orientation x,y,z,w. */
template <typename t_array>
inline void ODEBodyToX(const dBody &body, t_array x)
{
  const dReal *p= body.getPosition();  // x,y,z
  const dReal *q= body.getQuaternion();  // qw,qx,qy,qz
  x[0]= p[0]; x[1]= p[1]; x[2]= p[2];
  x[3]= q[1]; x[4]= q[2]; x[5]= q[3]; x[6]= q[0];
}
//-------------------------------------------------------------------------------------------



//===========================================================================================
// class TDynRobot
//===========================================================================================

void TDynRobot::Draw()
{
  dsSetTexture (DS_WOOD);

  dReal rad, len;
  dReal sides[4];
  for (std::vector<TNCBox>::const_iterator itr(link_b_.begin()),last(link_b_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    itr->getLengths(sides);
    dsDrawBox (itr->getPosition(), itr->getRotation(), sides);
  }
  for (std::vector<TNCSphere>::const_iterator itr(link_sp_.begin()),last(link_sp_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    dsDrawSphere (itr->getPosition(), itr->getRotation(), itr->getRadius());
  }
  for (std::vector<TNCCapsule>::const_iterator itr(link_ca_.begin()),last(link_ca_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    itr->getParams(&rad, &len);
    dsDrawCapsule (itr->getPosition(), itr->getRotation(), len,rad);
  }
  for (std::vector<TNCCylinder>::const_iterator itr(link_cy_.begin()),last(link_cy_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    itr->getParams(&rad, &len);
    dsDrawCylinder (itr->getPosition(), itr->getRotation(), len,rad);
  }
  for (std::vector<TTriMeshGeom>::const_iterator itr(link_tm_.begin()),last(link_tm_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    // const dReal *pos = itr->getPosition();
    // const dReal *rot = itr->getRotation();
    const dVector3 pos={0,0,0,0};
    const dMatrix3 rot={1,0,0,0 ,0,1,0,0, 0,0,1,0};

    for (int i(dGeomTriMeshGetTriangleCount(*itr)); i>0; --i)
    {
      std::cerr<<"i:"<<i<<std::endl;
      dVector3 v[3];
      dGeomTriMeshGetTriangle(*itr, i-1, &v[0], &v[1], &v[2]);
      std::cerr<<v[0][0]<<" "<<v[0][1]<<" "<<v[0][2]<<std::endl;
      std::cerr<<v[1][0]<<" "<<v[1][1]<<" "<<v[1][2]<<std::endl;
      std::cerr<<v[2][0]<<" "<<v[2][1]<<" "<<v[2][2]<<std::endl;
      dsDrawTriangle(pos, rot, v[0], v[1], v[2], 1);
    }
  }

  dVector3 pos,axis;
  dMatrix3 rot;
  for (std::vector<TNCHingeJoint>::const_iterator itr(joint_h_.begin()),last(joint_h_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    itr->getAnchor(pos);
    itr->getAxis(axis);
    dRFromAxisAndAngle (rot,-axis[1],axis[0],0.0, 0.5*M_PI-std::asin(axis[2]));
    dsDrawCylinder (pos, rot, VizJointLen,VizJointRad);
  }

  // for (std::vector<TNCSliderJoint>::const_iterator itr(joint_s_.begin()),last(joint_s_.end()); itr!=last; ++itr)
  // {
    // SetColor(itr->ColorCode);
    // // itr->getAnchor(pos);
    // const dReal *p1(dBodyGetPosition(itr->getBody(0))), *p2(dBodyGetPosition(itr->getBody(1)));
    // for(int d(0); d<3; ++d)  pos[d]= 0.5*(p1[d]+p2[d]);
    // itr->getAxis(axis);
    // dRFromAxisAndAngle (rot,-axis[1],axis[0],0.0, 0.5*M_PI-std::asin(axis[2]));
    // dsDrawCylinder (pos, rot, VizJointLen, VizJointRad);
  // }
}
//-------------------------------------------------------------------------------------------

//===========================================================================================
// class TJointChain1 : public TDynRobot
//===========================================================================================

/*override*/void TJointChain1::Create(dWorldID world, dSpaceID space)
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();
  link_tm_.clear();
  joint_b_.clear();
  joint_h_.clear();
  joint_h2_.clear();
  joint_s_.clear();
  joint_f_.clear();

  body_.resize(JointNum+1);
  link_cy_.resize(JointNum+1);

  double len= TotalLen/(double)(JointNum+1);

  int ib(0);
  for(int i(0); i<JointNum+1; ++i,++ib)
  {
    link_cy_[i].create(space,/*radius*/LinkRad,/*length*/len);
    body_[ib].create(world);
    body_[ib].setPosition(0.0, 0.0, (0.5+(double)i)*len);
    dMass m;
    // m.setCylinder(1.0,/*z*/3,/*radius*/LinkRad,/*length*/len);
    dMassSetCylinder(&m,1.0,/*z*/3,/*radius*/LinkRad,/*length*/len);
    body_[ib].setMass(&m);
    link_cy_[i].setBody(body_[ib]);
    link_cy_[i].ColorCode= i%6;
  }

  joint_h_.resize(JointNum);
  joint_f_.resize(1);

  int jh(0),jf(0);

  // base
  {
    joint_f_[jf].create(world);
    joint_f_[jf].attach(0,body_[0]);
    joint_f_[jf].set();
    ++jf;
  }

  dReal fmax(1000.0);  // NOTE: set zero to control by joint torque
  for(int j(0); j<JointNum; ++j,++jh)
  {
    joint_h_[j].create(world);
    joint_h_[j].attach(body_[j+1],body_[j]);
    joint_h_[j].setAnchor(0.0, 0.0, (double)(j+1)*len);
    switch(j%3)
    {
    case 0: joint_h_[j].setAxis(0.0,0.0,1.0); break;
    case 1: joint_h_[j].setAxis(1.0,0.0,0.0); break;
    case 2: joint_h_[j].setAxis(0.0,1.0,0.0); break;
    }
    joint_h_[j].setParam(dParamFMax,fmax);
  }
}
//-------------------------------------------------------------------------------------------


//===========================================================================================
// class TEnvironment
//===========================================================================================

void TEnvironment::Create()
{
  contactgroup_.create();
  world_.setGravity(0,0,Gravity);
  dWorldSetCFM(world_.id(),1e-5);
  plane_.create(space_,0,0,1,0);

  chain_.Create(world_,space_);
  // geom_.Create(world_,space_);

  time_= 0.0;
  sensors_.Clear();
  sensors_.SetZeros(JointNum);
}
//-------------------------------------------------------------------------------------------

void TEnvironment::StepSim(const double &time_step)
{
  ControlCallback(time_step);

  sensors_.ResetForStep();

  space_.collide(0,&NearCallback);
  world_.step(time_step);
  time_+= time_step;

  contactgroup_.empty();
}
//-------------------------------------------------------------------------------------------

void TEnvironment::Draw()
{
  if(Running)  EDrawCallback();

  chain_.Draw();
  // geom_.Draw();
}
//-------------------------------------------------------------------------------------------

void TEnvironment::ControlCallback(const double &time_step)
{
  dReal Kp(10.0);
  for(int j(0); j<JointNum; ++j)
    chain_.SetVelH(j, Kp*(TargetAngles[j]-chain_.GetAngleH(j)));
}
//-------------------------------------------------------------------------------------------

void TEnvironment::EDrawCallback()
{
  for(int j(0); j<JointNum; ++j)
    sensors_.JointAngles[j]= chain_.GetAngleH(j);

  for(int i(0); i<JointNum+1; ++i)
    ODEBodyToX(chain_.Body(i), sensors_.LinkX.begin()+7*i);

  sensors_.Time= time_;

  if(SensingCallback!=NULL)  SensingCallback(sensors_);
  if(DrawCallback!=NULL)  DrawCallback();
}
//-------------------------------------------------------------------------------------------

/* Called when b1 and b2 are colliding.
    Return whether we ignore this collision (true: ignore collision). */
bool TEnvironment::CollisionCallback(dBodyID &b1, dBodyID &b2, std::valarray<dContact> &contact)
{
  return false;
}
//-------------------------------------------------------------------------------------------


//===========================================================================================


static void NearCallback(void *data, dGeomID o1, dGeomID o2)
{
  assert(Env!=NULL);

  // do nothing if the two bodies are connected by a joint
  dBodyID b1= dGeomGetBody(o1);
  dBodyID b2= dGeomGetBody(o2);
  if(b1 && b2 && dAreConnected(b1,b2)) return;

  std::valarray<dContact> contact(MaxContacts);   // up to MaxContacts contacts per link
  for(int i(0); i<MaxContacts; ++i)
  {
    contact[i].surface.mode= dContactBounce | dContactSoftCFM;
    contact[i].surface.mu= 0.001; // dInfinity;
    contact[i].surface.mu2= 0.1;
    contact[i].surface.bounce= 0.1;
    contact[i].surface.bounce_vel= 0.01;
    contact[i].surface.soft_cfm= 0.1;
  }
  if(int numc=dCollide(o1,o2,MaxContacts,&contact[0].geom,sizeof(dContact)))
  {
    if(Env->CollisionCallback(b1,b2,contact))  return;  // ignore if the callback returns true
    for(int i(0); i<numc; ++i)
    {
      dJointID c= dJointCreateContact(Env->WorldID(),Env->ContactGroupID(),&contact[i]);
      dJointAttach (c,b1,b2);
    }
  }
}
//-------------------------------------------------------------------------------------------

static void SimStart()
{
  dAllocateODEDataForThread(dAllocateMaskAll);

  static float xyz[3] = {0.0,1.2447,0.7};
  static float hpr[3] = {-90.0000,-7.0000,0.0000};
  dsSetViewpoint (xyz,hpr);
}
//-------------------------------------------------------------------------------------------

static void SimLoop(int pause)
{
  assert(Env!=NULL);

  if (!pause && Running)
  {
    Env->StepSim(TimeStep);
  }

  Env->Draw();
  if(StepCallback!=NULL)  StepCallback(Env->Time(), TimeStep);
}
//-------------------------------------------------------------------------------------------

static void SimKeyevent(int command)
{
  assert(Env!=NULL);

  switch(command)
  {
  case 'r':
  case 'R': Create(); break;
  case ' ': Running= !Running; break;
  case 'z':
    for(int j(0); j<JointNum; ++j)  TargetAngles[j]+= 0.01;
    break;
  case 'x':
    for(int j(0); j<JointNum; ++j)  TargetAngles[j]-= 0.01;
    break;
  case 'n':
    std::cerr<<"Input number of joints > ";
    std::cin>>JointNum;
    std::cerr<<"New number ("<<JointNum<<") is effective after reset"<<std::endl;
    break;
  }
}
//-------------------------------------------------------------------------------------------

void Create()
{
  assert(Env!=NULL);
  TargetAngles.resize(JointNum);
  for(int j(0); j<JointNum; ++j)  TargetAngles[j]= 0.0;
  Env->Create();
}
//-------------------------------------------------------------------------------------------

void Reset()
{
  Create();
}
//-------------------------------------------------------------------------------------------

void Run(int argc, char **argv, const char *texture_path, int winx, int winy)
{
  // setup pointers to drawstuff callback functions
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &SimStart;
  fn.step = &SimLoop;
  fn.command = &SimKeyevent;
  fn.stop = 0;
  fn.path_to_textures = texture_path;

  dInitODE2(0);

  {
    TEnvironment env;
    Env= &env;
    // env.Create();
    Create();

    dsSimulationLoop (argc,argv,winx,winy,&fn);
  }  // env is deleted, which should be before dCloseODE()

  dCloseODE();
}
//-------------------------------------------------------------------------------------------

void Stop()
{
  dsStop();
}
//-------------------------------------------------------------------------------------------


}  // end of ode_x
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------

