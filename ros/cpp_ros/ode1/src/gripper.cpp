//-------------------------------------------------------------------------------------------
/*! \file    gripper.cpp
    \brief   ODE gripper simulator with virtual grasping joints
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.02, 2017
*/
//-------------------------------------------------------------------------------------------
#include "ode1/gripper.h"
#include "ode1/eiggeom.h"
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
// int    JointNum(7);
// bool   FixedBase(true);
// double TotalArmLen(1.0);
// double LinkRad(0.03);
double FSThick(0.001);
double FSSize(0.004);
double BaseLenX(0.08);
double BaseLenY(0.05);
double BaseLenZ(0.08);
double GBaseLenX(0.08);
double GBaseLenY(0.16);
double GBaseLenZ(0.05);
double GripLenY(0.02);
double GripLenZ(0.12);

int ObjectMode(2);  // ObjectMode: 0: None, 1: Box1, 2: Chair1 ...

double Box1PosX(0.7);
double Box1PosY(0.0);
double Box1SizeX(0.5);
double Box1SizeY(0.6);
double Box1SizeZ(0.4);
double Box1Density1(1.0);
double Box1Density2(50.0);

double Chair1PosX(0.6);
double Chair1PosY(0.0);
double Chair1BaseRad(0.2);
double Chair1BaseLen(0.10);
double Chair1Caster1Rad(0.02);
double Chair1Caster2Rad(0.02);
double Chair1Caster3Rad(0.02);
double Chair1Caster4Rad(0.02);
double Chair1CasterDX(0.1);
double Chair1CasterDY(0.1);
double Chair1Seat1Density(1.0);
double Chair1Seat1DX(0.0);
double Chair1Seat1DY(-0.05);
double Chair1Seat1SizeX(0.5);
double Chair1Seat1SizeY(0.6);
double Chair1Seat1SizeZ(0.12);
double Chair1Seat2Density(1.0);
double Chair1Seat2DX(0.0);
double Chair1Seat2DY(0.19);
double Chair1Seat2SizeX(0.5);
double Chair1Seat2SizeY(0.12);
double Chair1Seat2SizeZ(0.4);
double Chair1Damping(0.001);

double TimeStep(0.04);
double Gravity(-1.0);
bool   EnableKeyEvent(true);  // If we use a default key events.

double HingeFMax(100.0);
double SliderFMax(1.0);
int ControlMode(0);  // 0: Position, 1: Velocity, 2: Torque/Force
double _TargetBPose[]= {0.0,0.0,0.5, 0.0,0.0,0.0,1.0};
std::vector<double> TargetBPose(_TargetBPose,_TargetBPose+7);  // [x,y,z,quaternion(qx,qy,qz,qw)]
// std::vector<double> TargetAngles(JointNum,0.0);
std::vector<double> TargetGPos(2,0.0);
// std::vector<double> TargetVel(JointNum,0.0);
std::vector<double> TargetGVel(2,0.0);
// std::vector<double> TargetTorque(JointNum,0.0);
std::vector<double> TargetGForce(2,0.0);

bool Running(true);
void (*SensingCallback)(const TSensorsGr1 &sensors)= NULL;
void (*DrawCallback)(void)= NULL;
void (*StepCallback)(const double &time, const double &time_step)= NULL;
void (*KeyEventCallback)(int command)= NULL;
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

/* ODE::dBody velocity to velocity.
    v: [0-2]: linear velocity x,y,z, [3-5]: angular velocity x,y,z. */
template <typename t_array>
inline void ODEBodyToV(const dBody &body, t_array v)
{
  const dReal *p= body.getLinearVel();
  const dReal *q= body.getAngularVel();
  v[0]= p[0]; v[1]= p[1]; v[2]= p[2];
  v[3]= q[0]; v[4]= q[1]; v[5]= q[2];
}
//-------------------------------------------------------------------------------------------

/* ODE::dJointFeedback to force/torque.
    f: [0-2]: force, [3-5]: torque. */
template <typename t_array>
inline void ODEFeedbackToF(const dJointFeedback &feedback, t_array f)
{
  f[0]= feedback.f1[0]; f[1]= feedback.f1[1]; f[2]= feedback.f1[2];
  f[3]= feedback.t1[0]; f[4]= feedback.t1[1]; f[5]= feedback.t1[2];
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
// class TGripper1 : public TDynRobot
//===========================================================================================

/*override*/void TGripper1::Create(dWorldID world, dSpaceID space)
{
  clear();

  body_.resize(
      /*base,gripper:*/4
      + /*force sensors:*/3);
  link_b_.resize(/*base,gripper:*/4 + /*force sensors:*/3);
  int ib(0);  // body counter

  // create boxes
  dReal h_grip_begin= 0.5;
  dReal box_size_pos[7][6]={
      // base:
      /* 0*/{BaseLenX,BaseLenY,BaseLenZ,  0.0,0.0,h_grip_begin-FSThick-0.5*BaseLenZ},
      // gripper base, gripper-1, gripper-2:
      /* 1*/{GBaseLenX,GBaseLenY,GBaseLenZ, 0.0,0.0,h_grip_begin+0.5*GBaseLenZ},
      /* 2*/{GBaseLenX,GripLenY,GripLenZ,   0.0,+0.5*(GBaseLenY-GripLenY),h_grip_begin+GBaseLenZ+0.5*GripLenZ},
      /* 3*/{GBaseLenX,GripLenY,GripLenZ,   0.0,-0.5*(GBaseLenY-GripLenY),h_grip_begin+GBaseLenZ+0.5*GripLenZ},
      // force sensor between arm chain and gripper base:
      /* 4*/{FSSize,FSSize,FSThick,  0.0,0.0,h_grip_begin-0.5*FSThick},
      // force sensors on grippers:
      /* 5*/{GBaseLenX,FSThick,GripLenZ,   0.0,+(0.5*GBaseLenY-GripLenY-0.5*FSThick),h_grip_begin+GBaseLenZ+0.5*GripLenZ},
      /* 6*/{GBaseLenX,FSThick,GripLenZ,   0.0,-(0.5*GBaseLenY-GripLenY-0.5*FSThick),h_grip_begin+GBaseLenZ+0.5*GripLenZ}
    };
  for(int i(0); i<7; ++i,++ib)
  {
    dReal *size(box_size_pos[i]), *pos(box_size_pos[i]+3);
    body_[ib].create(world);
    body_[ib].setPosition(pos[0],pos[1],pos[2]);
    dMass m;
    m.setBox(1.0,size[0],size[1],size[2]);
    // m.adjust(mass);
    body_[ib].setMass(&m);
    link_b_[i].create(space,size[0],size[1],size[2]);
    link_b_[i].setBody(body_[ib]);
  }

  joint_f_.resize(/*fixing force sensors:*/4);
  joint_s_.resize(/*grippers:*/2);
  feedback_.resize(/*force sensors:*/3);
  int ifb(0);  // feedback counter

  dBodyID fixed_idxs[4][2]={
      // force sensor between base and gripper base:
      /* 0*/{body_[4],body_[0]},
      /* 1*/{body_[4],body_[1]},
      // force sensors on grippers:
      /* 2*/{body_[5],body_[2]},
      /* 3*/{body_[6],body_[3]},
    };
  for(int j(0); j<4; ++j)
  {
    joint_f_[j].create(world);
    joint_f_[j].attach(fixed_idxs[j][0],fixed_idxs[j][1]);
    joint_f_[j].set();
  }
  int fs_idxs[3]= {0, 2,3};
  for(int i(0); i<3; ++i,++ifb)
  {
    dJointSetFeedback(joint_f_[fs_idxs[i]], &feedback_[ifb]);
  }

  dReal fmaxs(SliderFMax);  // NOTE: set zero to control by joint torque
  if(ControlMode==2)  fmaxs= 0.0;  // Torque/Force control
  joint_s_[0].create(world);
  joint_s_[0].attach(body_[1],body_[2]);
  joint_s_[0].setAxis(0.0,+1.0,0.0);
  joint_s_[0].setParam(dParamFMax,fmaxs);

  joint_s_[1].create(world);
  joint_s_[1].attach(body_[1],body_[3]);
  joint_s_[1].setAxis(0.0,-1.0,0.0);
  joint_s_[1].setParam(dParamFMax,fmaxs);

  body_grasped1= NULL;
  body_grasped2= NULL;
}
//-------------------------------------------------------------------------------------------

void TGripper1::Grasp(dWorldID world, dBodyID &body1, dBodyID &body2)
{
  if(joint_f_gr_.size()==0)
  {
    joint_f_gr_.resize(2);
  }

  if(body1!=NULL && body_grasped1!=body1)
  {
    body_grasped1= body1;
    joint_f_gr_[0].create(world);
    joint_f_gr_[0].attach(body_[5],body_grasped1);
    joint_f_gr_[0].set();
  }
  if(body2!=NULL && body_grasped2!=body2)
  {
    body_grasped2= body2;
    joint_f_gr_[1].create(world);
    joint_f_gr_[1].attach(body_[6],body_grasped2);
    joint_f_gr_[1].set();
  }
}
//-------------------------------------------------------------------------------------------

void TGripper1::Release()
{
  joint_f_gr_.clear();
  body_grasped1= NULL;
  body_grasped2= NULL;
}
//-------------------------------------------------------------------------------------------


//===========================================================================================
// class TObjBox1 : public TDynRobot
//===========================================================================================

/*override*/void TObjBox1::Create(dWorldID world, dSpaceID space)
{
  clear();

  body_.resize(2);
  link_b_.resize(2);
  int ib(0);  // body counter

  {
    body_[ib].create(world);
    body_[ib].setPosition(Box1PosX,Box1PosY+0.25*Box1SizeY,0.5*Box1SizeZ);
    dMass m;
    m.setBox(Box1Density1,Box1SizeX,0.5*Box1SizeY,Box1SizeZ);
    // m.adjust(mass);
    body_[ib].setMass(&m);
    link_b_[0].create(space,Box1SizeX,0.5*Box1SizeY,Box1SizeZ);
    link_b_[0].setBody(body_[ib]);
    link_b_[0].ColorCode= 1;
    ++ib;
  }
  {
    body_[ib].create(world);
    body_[ib].setPosition(Box1PosX,Box1PosY-0.25*Box1SizeY,0.5*Box1SizeZ);
    dMass m;
    m.setBox(Box1Density2,Box1SizeX,0.5*Box1SizeY,Box1SizeZ);
    // m.adjust(mass);
    body_[ib].setMass(&m);
    link_b_[1].create(space,Box1SizeX,0.5*Box1SizeY,Box1SizeZ);
    link_b_[1].setBody(body_[ib]);
    link_b_[1].ColorCode= 2;
    ++ib;
  }

  joint_f_.resize(1);

  joint_f_[0].create(world);
  joint_f_[0].attach(body_[0],body_[1]);
  joint_f_[0].set();
}
//-------------------------------------------------------------------------------------------

//===========================================================================================
// class TObjChair1 : public TDynRobot
//===========================================================================================

template <typename t_in>
t_in MaxIn4(const t_in &x1, const t_in &x2, const t_in &x3, const t_in &x4)
{
  t_in xres(x1);
  if(x2>xres)  xres= x2;
  if(x3>xres)  xres= x3;
  if(x4>xres)  xres= x4;
  return xres;
}
//-------------------------------------------------------------------------------------------

/*override*/void TObjChair1::Create(dWorldID world, dSpaceID space)
{
  clear();

  body_.resize(/*base:*/1+/*seat:*/2+/*caster:*/4);
  link_cy_.resize(1);  // base
  link_b_.resize(2);  // seat
  link_sp_.resize(4);  // caster
  int ib(0);  // body counter

  double caster_rad_max= MaxIn4(Chair1Caster1Rad, Chair1Caster2Rad, Chair1Caster3Rad, Chair1Caster4Rad);
  {
    int i(0);
    link_cy_[i].create(space,/*radius*/Chair1BaseRad,/*length*/Chair1BaseLen);
    body_[ib].create(world);
    body_[ib].setPosition(Chair1PosX, Chair1PosY, 2.0*caster_rad_max+0.5*Chair1BaseLen);
    dMass m;
    // m.setCylinder(1.0,/*z*/3,/*radius*/Chair1BaseRad,/*length*/Chair1BaseLen);
    dMassSetCylinder(&m,1.0,/*z*/3,/*radius*/Chair1BaseRad,/*length*/Chair1BaseLen);
    body_[ib].setMass(&m);
    link_cy_[i].setBody(body_[ib]);
    ++ib;
  }

  dReal box_size_pos[2][7]={
      /* 0*/{Chair1Seat1Density, Chair1PosX+Chair1Seat1DX,Chair1PosY+Chair1Seat1DY,2.0*caster_rad_max+Chair1BaseLen+0.5*Chair1Seat1SizeZ,  Chair1Seat1SizeX,Chair1Seat1SizeY,Chair1Seat1SizeZ},
      /* 1*/{Chair1Seat2Density, Chair1PosX+Chair1Seat2DX,Chair1PosY+Chair1Seat2DY,2.0*caster_rad_max+Chair1BaseLen+Chair1Seat1SizeZ+0.5*Chair1Seat2SizeZ,  Chair1Seat2SizeX,Chair1Seat2SizeY,Chair1Seat2SizeZ}
    };
  for(int i(0); i<2; ++i,++ib)
  {
    dReal density(box_size_pos[i][0]), *pos(box_size_pos[i]+1), *size(box_size_pos[i]+4);
    body_[ib].create(world);
    body_[ib].setPosition(pos[0],pos[1],pos[2]);
    dMass m;
    m.setBox(1.0,size[0],size[1],size[2]);
    // m.adjust(mass);
    body_[ib].setMass(&m);
    link_b_[i].create(space,size[0],size[1],size[2]);
    link_b_[i].setBody(body_[ib]);
  }

  dReal sp_rad_pos[4][4]={
      /* 0*/{Chair1Caster1Rad, Chair1PosX+Chair1CasterDX,Chair1PosY+Chair1CasterDY,2.0*caster_rad_max-Chair1Caster1Rad},
      /* 1*/{Chair1Caster2Rad, Chair1PosX-Chair1CasterDX,Chair1PosY+Chair1CasterDY,2.0*caster_rad_max-Chair1Caster2Rad},
      /* 2*/{Chair1Caster3Rad, Chair1PosX-Chair1CasterDX,Chair1PosY-Chair1CasterDY,2.0*caster_rad_max-Chair1Caster3Rad},
      /* 3*/{Chair1Caster4Rad, Chair1PosX+Chair1CasterDX,Chair1PosY-Chair1CasterDY,2.0*caster_rad_max-Chair1Caster4Rad},
    };
  for(int i(0); i<4; ++i,++ib)
  {
    dReal *pos(sp_rad_pos[i]+1);
    link_sp_[i].create(space, sp_rad_pos[i][0]);
    body_[ib].create(world);
    body_[ib].setPosition(pos[0],pos[1],pos[2]);
    dMass m;
    m.setSphere(1.0, sp_rad_pos[i][0]);
    body_[ib].setMass(&m);
    link_sp_[i].setBody(body_[ib]);
  }

  joint_f_.resize(5);
  dBodyID fixed_idxs[5][2]={
      // casters:
      /* 0*/{body_[3],body_[0]},
      /* 1*/{body_[4],body_[0]},
      /* 2*/{body_[5],body_[0]},
      /* 3*/{body_[6],body_[0]},
      // seat:
      /* 4*/{body_[1],body_[2]}
    };
  for(int j(0),j_end(joint_f_.size()); j<j_end; ++j)
  {
    joint_f_[j].create(world);
    joint_f_[j].attach(fixed_idxs[j][0],fixed_idxs[j][1]);
    joint_f_[j].set();
  }

  joint_h_.resize(1);
  {
    int j(0);
    joint_h_[j].create(world);
    joint_h_[j].attach(body_[0],body_[1]);
    joint_h_[j].setAnchor(Chair1PosX, Chair1PosY, 2.0*caster_rad_max+Chair1BaseLen);
    joint_h_[j].setAxis(0.0,0.0,1.0);
    joint_h_[j].setParam(dParamFMax,0.0);
  }
}
//-------------------------------------------------------------------------------------------


//===========================================================================================
// class TObjDoor1 : public TDynRobot
//===========================================================================================
#if 0
/*override*/void TObjDoor1::Create(dWorldID world, dSpaceID space)
{
  clear();

  body_.resize(/*door:*/1+/*walls:*/2+/*stopper:*/1+/*knob:*/2);
  link_b_.resize(/*door:*/1+/*walls:*/2+/*stopper:*/1);
  link_cy_.resize(/*knob:*/2);
  int ib(0);  // body counter

double Door1X(0.7);
double Door1Y(0.0);
double Door1SizeX(0.05);
double Door1SizeY(0.7);
double Door1SizeZ(1.5);
double Door1Wall1DX(0.05);
double Door1Wall1DY(0.0);
double Door1Wall2DX(0.05);
double Door1Wall2DY(0.05);
double Door1WallSizeX(0.05);
double Door1WallSizeY(1.6);
double Door1WallSizeZ(1.6);

  dReal box_size_pos[4][6]={
      // door:
      /* 0*/{Door1X,Door1Y,0.5*Door1SizeZ,  Door1SizeX,Door1SizeY,Door1SizeZ},
      // wall-1, wall-2:
      /* 1*/{Door1X+Door1Wall1DX,Door1Y+Door1Wall1DY,0.5*Door1WallSizeZ, 0.0,0.0,h_grip_begin+0.5*GBaseLenZ},
      /* 2*/{GBaseLenX,GripLenY,GripLenZ,   0.0,+0.5*(GBaseLenY-GripLenY),h_grip_begin+GBaseLenZ+0.5*GripLenZ},
      // stopper:
      /* 3*/{GBaseLenX,GripLenY,GripLenZ,   0.0,-0.5*(GBaseLenY-GripLenY),h_grip_begin+GBaseLenZ+0.5*GripLenZ},
    };
  for(int i(0); i<12; ++i,++ib)
  {
    dReal *size(box_size_pos[i]), *pos(box_size_pos[i]+3);
    body_[ib].create(world);
    body_[ib].setPosition(pos[0],pos[1],pos[2]);
    dMass m;
    m.setBox(1.0,size[0],size[1],size[2]);
    // m.adjust(mass);
    body_[ib].setMass(&m);
    link_b_[i].create(space,size[0],size[1],size[2]);
    link_b_[i].setBody(body_[ib]);
  }

  joint_f_.resize(1);

  joint_f_[0].create(world);
  joint_f_[0].attach(body_[0],body_[1]);
  joint_f_[0].set();
}
//-------------------------------------------------------------------------------------------
#endif


//===========================================================================================
// class TEnvironment
//===========================================================================================

void TEnvironment::Create()
{
  contactgroup_.create();
  world_.setGravity(0,0,Gravity);
  dWorldSetCFM(world_.id(),1e-5);
  plane_.create(space_,0,0,1,0);

  gripper_.Create(world_,space_);
  // ObjectMode: 0: None, 1: Box1, 2: Chair1 ...
  if(ObjectMode==1)  box1_.Create(world_,space_);
  if(ObjectMode==2)  chair1_.Create(world_,space_);
  // geom_.Create(world_,space_);

  time_= 0.0;
  sensors_.Clear();
  sensors_.SetZeros();
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

  gripper_.Draw();
  // ObjectMode: 0: None, 1: Box1, 2: Chair1 ...
  if(ObjectMode==1)  box1_.Draw();
  if(ObjectMode==2)  chair1_.Draw();
  // geom_.Draw();
}
//-------------------------------------------------------------------------------------------

void TEnvironment::ControlCallback(const double &time_step)
{
  {
    TEVector7 x1, x2;  // Poses of force sensors
    double xb1[7], xb2[7];
    dBody &body1(gripper_.Body(5));  // Force sensors
    ODEBodyToX(body1, xb1);
    x1= ToEVec(xb1);
    dBody &body2(gripper_.Body(6));  // Force sensors
    ODEBodyToX(body2, xb2);
    x2= ToEVec(xb2);

    TEVector3 fl1,fl2;  // Forces on fingertips in local frame.
    double f1[6],f2[6];  // Forces and torques on fingertips.
    ODEFeedbackToF(gripper_.GetFeedback(1), f1);
    ODEFeedbackToF(gripper_.GetFeedback(2), f2);
    fl1= (XToQ(x1).inverse()*XToPos(ToEVec3(f1))).translation();
    fl2= (XToQ(x2).inverse()*XToPos(ToEVec3(f2))).translation();
    //*DEBUG*/std::cerr<< ToEVec3(f1).transpose() <<", "<< ToEVec3(f2).transpose() <<std::endl;
    /*DEBUG*/std::cerr<< fl1.transpose() <<", "<< fl2.transpose() <<std::endl;

    if(fl1[1]<-0.1 && fl2[1]>0.1)
    {
      gripper_.Grasp(world_, sensors_.FingerColliding1, sensors_.FingerColliding2);
      /*DEBUG*/std::cerr<<"grasped"<<std::endl;
    }
    if(fl1[1]>0.2 || fl2[1]<-0.2)
    {
      gripper_.Release();
    }
  }

  // 0: Position, 1: Velocity, 2: Torque/Force
  if(ControlMode==0)  // Position control
  {
    dReal Kps(10.0);
    for(int j(0); j<2; ++j)
      gripper_.SetVelS(j, Kps*(TargetGPos[j]-gripper_.GetPosS(j)));

    dReal Kpbf(0.1), Kdbf(0.02);
    dReal Kpbt(0.0005), Kdbt(0.00005);
    double xb[7], vb[7];
    TEVector7 x, x_trg;
    TEVector6 dx;
    dBody &body(gripper_.Body(0));  // Gripper base
    ODEBodyToX(body, xb);
    ODEBodyToV(body, vb);
    x= ToEVec(xb);
    x_trg= ToEVec(TargetBPose.begin());
    dx= DiffX(x, x_trg);
    // std::cerr<<x_trg.transpose()<<" / "<<x.transpose()<<" / "<<dx.transpose()<<std::endl;
    body.addForce(Kpbf*dx[0]-Kdbf*vb[0],Kpbf*dx[1]-Kdbf*vb[1],Kpbf*dx[2]-Kdbf*vb[2]);
    body.addTorque(Kpbt*dx[3]-Kdbt*vb[3],Kpbt*dx[4]-Kdbt*vb[4],Kpbt*dx[5]-Kdbt*vb[5]);
  }
  else if(ControlMode==1)  // Velocity control
  {
    for(int j(0); j<2; ++j)
      gripper_.SetVelS(j, TargetGVel[j]);
  }
  else if(ControlMode==2)  // Torque/Force control
  {
    for(int j(0); j<2; ++j)
      gripper_.AddForceS(j, TargetGForce[j]);
  }
  /*TEST of getForce: it gives the force added by user (not actually simulated force); i.e. useless in general.
  dVector3 f;
  int i=1;
  std::cerr<<"F= "<<gripper_.Body(i).getForce()[0]<<", "<<gripper_.Body(i).getForce()[1]<<", "<<gripper_.Body(i).getForce()[2]<<std::endl;
  //*/
  /*TEST of dJointSetFeedback
  std::cerr<<"F= "<<joint_feedback1.f1[0]<<", "<<joint_feedback1.f1[1]<<", "<<joint_feedback1.f1[2]<<std::endl;
  std::cerr<<"T= "<<joint_feedback1.t1[0]<<", "<<joint_feedback1.t1[1]<<", "<<joint_feedback1.t1[2]<<std::endl;
  //*/

  // ObjectMode: 0: None, 1: Box1, 2: Chair1 ...
  if(ObjectMode==2)
  {
    chair1_.AddTorqueH(0, -Chair1Damping*chair1_.GetAngVelH(0));
  }

}
//-------------------------------------------------------------------------------------------

void TEnvironment::EDrawCallback()
{
  // for(int j(0); j<JointNum; ++j)
    // sensors_.JointAngles[j]= gripper_.GetAngleH(j);

  for(int i(0); i<7; ++i)
    ODEBodyToX(gripper_.Body(i), sensors_.LinkX.begin()+7*i);

  for(int i(0); i<3; ++i)
    ODEFeedbackToF(gripper_.GetFeedback(i), sensors_.Forces.begin()+6*i);

  for(int i(0); i<7; ++i)
    sensors_.Masses[i]= gripper_.Body(i).getMass().mass;

  // ObjectMode: 0: None, 1: Box1, 2: Chair1 ...
  if(ObjectMode==1)
  {
    ODEBodyToX(box1_.Body(0), sensors_.Box1X.begin());
    sensors_.Box1X[0]= 0.5*(sensors_.Box1X[0]+box1_.Body(1).getPosition()[0]);
    sensors_.Box1X[1]= 0.5*(sensors_.Box1X[1]+box1_.Body(1).getPosition()[1]);
    sensors_.Box1X[2]= 0.5*(sensors_.Box1X[2]+box1_.Body(1).getPosition()[2]);
  }
  if(ObjectMode==2)
  {
    for(int i(0); i<3; ++i)
      ODEBodyToX(chair1_.Body(i), sensors_.Chair1X.begin()+7*i);
  }

  sensors_.Time= time_;

  if(SensingCallback!=NULL)  SensingCallback(sensors_);
  if(DrawCallback!=NULL)  DrawCallback();
}
//-------------------------------------------------------------------------------------------

/* Called when b1 and b2 are colliding.
    Return whether we ignore this collision (true: ignore collision). */
bool TEnvironment::CollisionCallback(dBodyID &b1, dBodyID &b2, std::valarray<dContact> &contact)
{
  dBodyID finger1(gripper_.Body(5));  // Force sensors
  dBodyID finger2(gripper_.Body(6));  // Force sensors
  if(b1==finger1)  sensors_.FingerColliding1= b2;
  if(b2==finger1)  sensors_.FingerColliding1= b1;
  if(b1==finger2)  sensors_.FingerColliding2= b2;
  if(b2==finger2)  sensors_.FingerColliding2= b1;

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

  if(KeyEventCallback!=NULL)  KeyEventCallback(command);
  if(EnableKeyEvent)
  {
    TEVector7 x_trg= ToEVec(TargetBPose.begin());
    switch(command)
    {
    case 'r':
    case 'R': Create(); break;
    case ' ': Running= !Running; break;
    case 'w': x_trg= Transform(TEVector3(0.05,0.0,0.0),x_trg);   for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 's': x_trg= Transform(TEVector3(-0.05,0.0,0.0),x_trg);  for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 'a': x_trg= Transform(TEVector3(0.0,0.05,0.0),x_trg);   for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 'd': x_trg= Transform(TEVector3(0.0,-0.05,0.0),x_trg);  for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 'z': x_trg= Transform(TEVector3(0.0,0.0,0.05),x_trg);   for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 'x': x_trg= Transform(TEVector3(0.0,0.0,-0.05),x_trg);  for(int i(0);i<3;++i)TargetBPose[i]=x_trg[i];  break;
    case 'W': x_trg= Transform(QFromAxisAngle(TEVector3(0.0,1.0,0.0),0.1),x_trg);    for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case 'S': x_trg= Transform(QFromAxisAngle(TEVector3(0.0,1.0,0.0),-0.1),x_trg);   for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case 'A': x_trg= Transform(QFromAxisAngle(TEVector3(1.0,0.0,0.0),0.1),x_trg);    for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case 'D': x_trg= Transform(QFromAxisAngle(TEVector3(1.0,0.0,0.0),-0.1),x_trg);   for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case 'Z': x_trg= Transform(QFromAxisAngle(TEVector3(0.0,0.0,1.0),0.1),x_trg);    for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case 'X': x_trg= Transform(QFromAxisAngle(TEVector3(0.0,0.0,1.0),-0.1),x_trg);   for(int i(3);i<7;++i)TargetBPose[i]=x_trg[i];  break;
    case ']':  TargetGPos[0]+= 0.002; TargetGPos[1]+= 0.002;  break;
    case '[':  TargetGPos[0]-= 0.002; TargetGPos[1]-= 0.002;  break;
    // case 'n':
      // std::cerr<<"Input number of joints > ";
      // std::cin>>JointNum;
      // std::cerr<<"New number ("<<JointNum<<") is effective after reset"<<std::endl;
      // break;
    }
  }
}
//-------------------------------------------------------------------------------------------

void Create()
{
  assert(Env!=NULL);
  // TargetAngles.resize(JointNum);
  // TargetVel.resize(JointNum);
  // TargetTorque.resize(JointNum);
  // for(int j(0); j<JointNum; ++j)
  // {
    // TargetAngles[j]= 0.0;
    // TargetVel[j]= 0.0;
    // TargetTorque[j]= 0.0;
  // }
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
