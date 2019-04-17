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

namespace ode_test
{

class TEnvironment;
static TEnvironment *Env(NULL);
static const int MAX_CONTACTS(10);  // maximum number of contact points per body
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
DEF_NC(BallJoint)
DEF_NC(Box)
#undef DEF_NC

class TDynRobot
{
public:

  std::vector<TNCBody>& Body() {return body_;}
  dBody& Body(int j) {return body_[j];}

  void Create(dWorldID world, dSpaceID space);
  void Draw();

private:
  std::vector<TNCBody> body_;
  std::vector<TNCBallJoint> joint_b_;
  std::vector<TNCBox> link_b_;
};

void TDynRobot::Create(dWorldID world, dSpaceID space)
{
  dReal side(0.2),mass(1.0);

  body_.resize(10);
  link_b_.resize(10);
  for (int i=0; i<10; i++)
  {
    body_[i].create (world);
    dReal k = i*side;
    body_[i].setPosition (k,k,k+0.4);
    dMass m;
    m.setBox (1,side,side,side);
    m.adjust (mass);
    body_[i].setMass (&m);
    body_[i].setData ((void*)(size_t)i);

    link_b_[i].create (space,side,side,side);
    link_b_[i].setBody (body_[i]);
  }

  joint_b_.resize(9);
  for (int i=0; i<9; i++)
  {
    joint_b_[i].create (world);
    joint_b_[i].attach (body_[i],body_[i+1]);
    dReal k = (i+0.5)*side;
    joint_b_[i].setAnchor (k,k,k+0.4);
  }
}

void TDynRobot::Draw()
{
  dReal sides[3] = {0.2,0.2,0.2};
  dsSetColor (1,1,0);
  dsSetTexture (DS_WOOD);
  for (size_t i=0; i<body_.size(); i++)
    dsDrawBox (body_[i].getPosition(),body_[i].getRotation(),sides);
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
  robot_.Body().back().addForce (0,0,1.5*(std::sin(angle)+1.0));

  space_.collide (0,&NearCallback);
  world_.step (0.05);

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

  static float xyz[3] = {2.1640f,-1.3079f,1.7600f};
  static float hpr[3] = {125.5000f,-17.0000f,0.0000f};
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

  dsSimulationLoop (argc,argv,400,300,&fn);

  dCloseODE();
  return 0;
}
