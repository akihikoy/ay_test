#include <ode/ode.h>
#include <drawstuff/drawstuff.h>

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#endif

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

namespace ode_test
{

class TEnvironment;
static TEnvironment *Env(NULL);
static const int MAX_CONTACTS(10);  // maximum number of contact points per body
void NearCallback(void*,dGeomID,dGeomID);
dReal JOINT_LEN(0.3),JOINT_RAD(0.1);

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
DEF_NC(FixedJoint)
DEF_NC(Box)
DEF_NC(Capsule)
DEF_NC(Cylinder)
DEF_NC(Sphere)
#undef DEF_NC

class TDynRobot
{
public:

  std::vector<TNCBody>& Body() {return body_;}
  dBody& Body(int j) {return body_[j];}

  void Create(dWorldID world, dSpaceID space);
  void Draw();
  void SetVelH(int j, dReal vel)  {joint_h_[j].setParam(dParamVel,vel);}
  void GetHandPos(dReal *pos);

private:
  std::vector<TNCBody> body_;
  std::vector<TNCBox> link_b_;
  std::vector<TNCCapsule> link_ca_;
  std::vector<TNCCylinder> link_cy_;
  std::vector<TNCSphere> link_sp_;
  std::vector<TNCHingeJoint> joint_h_;
  std::vector<TNCHinge2Joint> joint_h2_;
  std::vector<TNCFixedJoint> joint_f_;
};

void TDynRobot::Create(dWorldID world, dSpaceID space)
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();

  body_.resize(5);
  link_b_.resize(1);
  link_ca_.resize(2);
  link_cy_.resize(2);

  int i(0),ib(0),ica(0),icy(0);

  // base1
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.05);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*z*/3,/*radius*/0.3,/*length*/0.1);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.3,/*length*/0.1);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }
  // base2
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.15);
    dMass m;
    // m.setCylinder (1,/*z*/3,/*radius*/0.3,/*length*/0.2);
    dMassSetCylinder (&m,1,/*z*/3,/*radius*/0.3,/*length*/0.1);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_cy_[icy].create (space,/*radius*/0.3,/*length*/0.1);
    link_cy_[icy].setBody (body_[i]);
    ++i;++icy;
  }
  // arm1
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.4);
    dMass m;
    m.setCapsule (1,/*z*/3,/*radius*/0.1,/*length*/0.4-0.1*2.0);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_ca_[ica].create (space,/*radius*/0.1,/*length*/0.4-0.1*2.0);
    link_ca_[ica].setBody (body_[i]);
    ++i;++ica;
  }
  // arm2
  {
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,0.8);
    dMass m;
    m.setCapsule (1,/*z*/3,/*radius*/0.1,/*length*/0.4-0.1*2.0);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_ca_[ica].create (space,/*radius*/0.1,/*length*/0.4-0.1*2.0);
    link_ca_[ica].setBody (body_[i]);
    ++i;++ica;
  }
  // hand (fixed to arm2)
  {
    dReal side(0.25),side2(0.2);
    body_[i].create (world);
    body_[i].setPosition (0.0,0.0,1.0);
    dMass m;
    m.setBox (1,side,side,side2);
    // m.adjust (mass);
    body_[i].setMass (&m);
    link_b_[ib].create (space,side,side,side2);
    link_b_[ib].setBody (body_[i]);
    ++i;++ib;
  }

  dReal fmax(1000.0);

  joint_h_.clear();
  joint_h2_.clear();
  joint_f_.clear();

  joint_h_.resize(3);
  joint_f_.resize(2);

  int jh(0),jf(0);

  // base1
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (0,body_[0]);
    joint_f_[jf].set();
    ++jf;
  }
  // base2
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[0],body_[1]);
    joint_h_[jh].setAnchor (0.0,0.0,0.1);
    joint_h_[jh].setAxis (0.0,0.0,1.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // arm1
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[1],body_[2]);
    joint_h_[jh].setAnchor (0.0,0.0,0.2);
    joint_h_[jh].setAxis (0.0,1.0,0.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // arm2
  {
    joint_h_[jh].create (world);
    joint_h_[jh].attach (body_[2],body_[3]);
    joint_h_[jh].setAnchor (0.0,0.0,0.6);
    joint_h_[jh].setAxis (0.0,1.0,0.0);
    joint_h_[jh].setParam(dParamFMax,fmax);
    ++jh;
  }
  // hand (fixed to arm2)
  {
    joint_f_[jf].create (world);
    joint_f_[jf].attach (body_[3],body_[4]);
    joint_f_[jf].set();
    ++jf;
  }
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
    dsDrawCylinder (pos, rot, JOINT_LEN,JOINT_RAD);
  }
}

class TEnvironment
{
public:
  TEnvironment()
    : space_(0) {}

  dWorldID WorldID() {return world_.id();}
  dSpaceID SpaceID() {return space_.id();}
  dJointGroupID ContactGroupID() {return contactgroup_.id();}

  TDynRobot& Robot() {return robot_;}

  void Create();
  void StepSim(const double &time_step);
  void Draw();

private:
  dWorld world_;
  dSimpleSpace space_;
  dJointGroup contactgroup_;

  TDynRobot robot_;
  dPlane    plane_;
};

void TEnvironment::Create()
{
  contactgroup_.create (0);
  world_.setGravity (0,0,-0.5);
  dWorldSetCFM (world_.id(),1e-5);
  plane_.create (space_,0,0,1,0);

  robot_.Create(world_,space_);
}

void TEnvironment::StepSim(const double &time_step)
{
  static double angle = 0;
  angle += 0.05;
//   robot_.Body().back().addForce (0.1*(std::sin(angle)+1.0), 0.1*(std::sin(angle*1.7)+1.0) ,0.1*(std::sin(angle*0.7)+1.0));

  robot_.SetVelH(0,1.5);
  robot_.SetVelH(1,0.1);
  robot_.SetVelH(2,0.0);

  space_.collide (0,&NearCallback);
  world_.step (time_step);

  contactgroup_.empty();
}

void TEnvironment::Draw()
{
  robot_.Draw();
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
// #define DBG(x) std::cerr<<#x"= "<<x<<std::endl;
// #define DBG(x) x<<"\t"<<
// std::cout<<
// DBG(contact[i].surface.mode)
// DBG(contact[i].surface.mu)
// DBG(contact[i].surface.mu2)
// DBG(contact[i].surface.bounce)
// DBG(contact[i].surface.bounce_vel)
// DBG(contact[i].surface.soft_erp)
// DBG(contact[i].surface.soft_cfm)
// DBG(contact[i].surface.motion1)
// DBG(contact[i].surface.motion2)
// DBG(contact[i].surface.motionN)
// DBG(contact[i].surface.slip1)
// DBG(contact[i].surface.slip2)
// std::endl;
// #undef DBG

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
    Env->StepSim(0.05);
  }

  Env->Draw();
}

} // end of ode_test

int main (int argc, char **argv)
{
  // setup pointers to drawstuff callback functions
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &ode_test::SimStart;
  fn.step = &ode_test::SimLoop;
  fn.command = 0;
  fn.stop = 0;
  fn.path_to_textures = "textures";

  dInitODE2(0);

  ode_test::TEnvironment env;
  ode_test::Env= &env;
  env.Create();

  dsSimulationLoop (argc,argv,400,400,&fn);

  dCloseODE();
  return 0;
}
