#ifdef xode_h
#  error DO NOT INCLUDE xode.h TWICE
#endif

#define xode_h

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawLine dsDrawLineD
#endif

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace xode
{

class TEnvironment;
class TDynRobot;
TEnvironment *Env(NULL);
const int MAX_CONTACTS(10);  // maximum number of contact points per body
dReal JointLen(0.3), JointRad(0.1);
void (*ControlCallback)(TEnvironment &env, TDynRobot &robot)= NULL;
void (*DrawCallback)(TEnvironment &env, TDynRobot &robot)= NULL;
void (*KeyEventCallback)(TEnvironment &env, TDynRobot &robot, int command)= NULL;

void NearCallback(void*,dGeomID,dGeomID);


// the following classes have a copy constructor and operator= that do nothing;
// these classes are defined in order to use std::vector of them
#define DEF_NC(x_class) \
  class TNC##x_class : public d##x_class  \
  {                                       \
  public:                                 \
    TNC##x_class() : d##x_class(){}       \
    TNC##x_class(const TNC##x_class&)     \
      : d##x_class(){}                    \
    const TNC##x_class& operator=(const TNC##x_class&) {return *this;} \
  private:                                \
  };
DEF_NC(Body)
DEF_NC(HingeJoint)
DEF_NC(Hinge2Joint)
DEF_NC(UniversalJoint)
DEF_NC(FixedJoint)
DEF_NC(Box)
DEF_NC(Capsule)
DEF_NC(Cylinder)
DEF_NC(Sphere)
#undef DEF_NC

class TDynRobot
{
public:

  TDynRobot() {}
  ~TDynRobot() {}

  std::vector<TNCBody>& Body() {return body_;}
  dBody& Body(int j) {return body_[j];}

  void Clear();

  void Create(dWorldID world, dSpaceID space);
  void Draw();

  const dReal GetAngleHinge(int j)  {return joint_h_[j].getAngle();}
  const dReal GetAngVelHinge(int j)  {return joint_h_[j].getAngleRate();}

  const dReal GetAngleUniversal1(int j)  {return joint_u_[j].getAngle1();}
  const dReal GetAngVelUniversal1(int j)  {return joint_u_[j].getAngle1Rate();}
  const dReal GetAngleUniversal2(int j)  {return joint_u_[j].getAngle2();}
  const dReal GetAngVelUniversal2(int j)  {return joint_u_[j].getAngle2Rate();}

  void SetAngVelHinge(int j, const dReal &vel)  {joint_h_[j].setParam(dParamVel,vel);}
  void AddTorqueHinge(int j, const dReal &u)  {joint_h_[j].addTorque(u);}

  void SetAngVelUniversal1(int j, const dReal &vel)  {joint_u_[j].setParam(dParamVel1,vel);}
  void SetAngVelUniversal2(int j, const dReal &vel)  {joint_u_[j].setParam(dParamVel2,vel);}
  void AddTorquesUniversal(int j, const dReal &u1, const dReal &u2)  {joint_u_[j].addTorques(u1,u2);}

  const dReal* GetHandPos();
  const dReal* GetHandRot();

private:
  std::vector<TNCBody> body_;
  std::vector<TNCBox> link_b_;
  std::vector<TNCCapsule> link_ca_;
  std::vector<TNCCylinder> link_cy_;
  std::vector<TNCSphere> link_sp_;
  std::vector<TNCHingeJoint> joint_h_;
  std::vector<TNCHinge2Joint> joint_h2_;
  std::vector<TNCUniversalJoint> joint_u_;
  std::vector<TNCFixedJoint> joint_f_;
};

void TDynRobot::Clear()
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();

  joint_h_.clear();
  joint_h2_.clear();
  joint_f_.clear();
}

void TDynRobot::Draw()
{
  dsSetTexture (DS_WOOD);

  dReal rad, len;
  dReal sides[4];
  dsSetColorAlpha (0.0, 1.0, 0.0, 0.8);
  for (std::vector<TNCBox>::const_iterator itr(link_b_.begin()),last(link_b_.end()); itr!=last; ++itr)
  {
    itr->getLengths(sides);
    dsDrawBox (itr->getPosition(), itr->getRotation(), sides);
  }
  dsSetColorAlpha (0.0, 0.5, 1.0, 0.6);
  for (std::vector<TNCCapsule>::const_iterator itr(link_ca_.begin()),last(link_ca_.end()); itr!=last; ++itr)
  {
    itr->getParams(&rad, &len);
    dsDrawCapsule (itr->getPosition(), itr->getRotation(), len,rad);
  }
  dsSetColorAlpha (1.0, 0.0, 0.0, 0.6);
  for (std::vector<TNCCylinder>::const_iterator itr(link_cy_.begin()),last(link_cy_.end()); itr!=last; ++itr)
  {
    itr->getParams(&rad, &len);
    dsDrawCylinder (itr->getPosition(), itr->getRotation(), len,rad);
  }
  dsSetColorAlpha (0.0, 1.0, 0.5, 0.6);
  for (std::vector<TNCSphere>::const_iterator itr(link_sp_.begin()),last(link_sp_.end()); itr!=last; ++itr)
  {
    dsDrawSphere (itr->getPosition(), itr->getRotation(), itr->getRadius());
  }

  dVector3 pos,axis;
  dMatrix3 rot;
  dsSetColorAlpha (1.0, 1.0, 1.0, 0.8);
  for (std::vector<TNCHingeJoint>::const_iterator itr(joint_h_.begin()),last(joint_h_.end()); itr!=last; ++itr)
  {
    itr->getAnchor(pos);
    itr->getAxis(axis);
    dRFromAxisAndAngle (rot,-axis[1],axis[0],0.0, 0.5*M_PI-std::asin(axis[2]));
    dsDrawCylinder (pos, rot, JointLen, JointRad);
  }
  for (std::vector<TNCUniversalJoint>::const_iterator itr(joint_u_.begin()),last(joint_u_.end()); itr!=last; ++itr)
  {
    itr->getAnchor(pos);
    itr->getAxis1(axis);
    dRFromAxisAndAngle (rot,-axis[1],axis[0],0.0, 0.5*M_PI-std::asin(axis[2]));
    dsDrawCylinder (pos, rot, JointLen, JointRad);
    itr->getAxis2(axis);
    dRFromAxisAndAngle (rot,-axis[1],axis[0],0.0, 0.5*M_PI-std::asin(axis[2]));
    dsDrawCylinder (pos, rot, JointLen, JointRad);
  }
}

class TEnvironment
{
public:
  TEnvironment()
    : space_(0), time_(0.0), time_step_(0.05) {}

  dWorldID WorldID() {return world_.id();}
  dSpaceID SpaceID() {return space_.id();}
  dJointGroupID ContactGroupID() {return contactgroup_.id();}

  TDynRobot& Robot() {return robot_;}

  const double& Time() const {return time_;}

  const double& TimeStep() const {return time_step_;}
  void TimeStep(const double &ts)  {time_step_=ts;}

  void Create();
  void StepSim();
  void Draw();
  void KeyEvent(int command);

private:
  dWorld world_;
  dSimpleSpace space_;
  dJointGroup contactgroup_;

  TDynRobot robot_;
  dPlane    plane_;

  double time_,time_step_;
};

void TEnvironment::Create()
{
  contactgroup_.create ();
  world_.setGravity (0,0,-0.5);
  dWorldSetCFM (world_.id(),1e-5);
  plane_.create (space_,0,0,1,0);

  robot_.Create(world_,space_);

  time_= 0.0;
}

void TEnvironment::StepSim()
{
  if(ControlCallback)  ControlCallback(*this,robot_);

  space_.collide (0,&NearCallback);
  world_.step (time_step_);
  time_+= time_step_;

  contactgroup_.empty();
}

void TEnvironment::Draw()
{
  robot_.Draw();
  if(DrawCallback)  DrawCallback(*this,robot_);
}

void TEnvironment::KeyEvent(int command)
{
  if(KeyEventCallback)  KeyEventCallback(*this,robot_,command);
}

void NearCallback(void *data, dGeomID o1, dGeomID o2)
{
  assert(Env!=NULL);

  // do nothing if the two bodies are connected by a joint
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnected (b1,b2)) return;

  dContact contact[MAX_CONTACTS];   // up to MAX_CONTACTS contacts per link
  for (int i=0; i<MAX_CONTACTS; i++)
  {
    contact[i].surface.mode = dContactBounce | dContactSoftCFM;
    contact[i].surface.mu = dInfinity;
    contact[i].surface.mu2 = 0;
    contact[i].surface.bounce = 0.1;
    contact[i].surface.bounce_vel = 0.1;
    contact[i].surface.soft_cfm = 0.01;
  }
  if (int numc=dCollide (o1,o2,MAX_CONTACTS,&contact[0].geom,sizeof(dContact)))
  {
    for (int i=0; i<numc; i++)
    {
      dJointID c= dJointCreateContact(Env->WorldID(),Env->ContactGroupID(),contact+i);
      dJointAttach (c,b1,b2);
    }
  }
}

void SimStart()
{
  dAllocateODEDataForThread(dAllocateMaskAll);

  static float xyz[3] = {0.6667,1.7789,1.3300};
  static float hpr[3] = {-104.0000,-29.5000,0.0000};
  dsSetViewpoint (xyz,hpr);
}

void SimLoop (int pause)
{
  assert(Env!=NULL);

  if (!pause)
  {
    Env->StepSim();
  }

  Env->Draw();
}

void KeyEvent (int command)
{
  assert(Env!=NULL);

  Env->KeyEvent(command);
}

inline dsFunctions SimInit(const char *path_to_textures, TEnvironment &env)
{
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &SimStart;
  fn.step = &SimLoop;
  fn.command = &KeyEvent;
  fn.stop = 0;
  fn.path_to_textures = path_to_textures;

  dInitODE2(0);

  Env= &env;
  env.Create();

  return fn;
}

} // end of xode
