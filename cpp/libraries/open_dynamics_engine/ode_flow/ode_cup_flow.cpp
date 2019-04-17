#include <ode/ode.h>
#include <drawstuff/drawstuff.h>

#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawTriangle dsDrawTriangleD
#endif

#include <vector>
#include <valarray>
#include <cassert>
#include <cmath>
#include <iostream>

#define override

namespace ode_test
{

class TEnvironment;
static TEnvironment *Env(NULL);
// static const int MAX_CONTACTS(10);  // maximum number of contact points per body
static const int MAX_CONTACTS(1);  // maximum number of contact points per body
static void NearCallback(void*,dGeomID,dGeomID);
dReal VIZ_JOINT_LEN(0.1),VIZ_JOINT_RAD(0.02);

dReal TargetAngle(0.0);
static const dReal BALL_RAD(0.025);
static const dReal BOX_THICKNESS(0.02);


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
          {0.0, 1.0, 1.0, 0.3}};
inline void SetColor(int i)
{
  dReal *col= IndexedColors[i];
  dsSetColorAlpha(col[0],col[1],col[2],col[3]);
}


// the following classes have a copy constructor and operator= that do nothing;
// these classes are defined in order to use std::vector of them
#define DEF_NC(x_class, x_col) \
  class TNC##x_class : public d##x_class  \
  {                                       \
  public:                                 \
    int ColorCode;                        \
    TNC##x_class() : d##x_class(), ColorCode(x_col) {} \
    TNC##x_class(const TNC##x_class&)     \
      : d##x_class(), ColorCode(x_col){}  \
    const TNC##x_class& operator=(const TNC##x_class&) {return *this;} \
  private:                                \
  };
DEF_NC(Body,0)
DEF_NC(BallJoint,0)
DEF_NC(HingeJoint,1)
DEF_NC(Hinge2Joint,2)
DEF_NC(FixedJoint,3)
DEF_NC(Box,0)
DEF_NC(Capsule,1)
DEF_NC(Cylinder,2)
DEF_NC(Sphere,3)
#undef DEF_NC

class TTriMeshGeom : public dGeom
{
public:
  int ColorCode;
  TTriMeshGeom() : dGeom(), ColorCode(4) {}

  TTriMeshGeom(const TTriMeshGeom&) : dGeom(), ColorCode(4) {}
  const TTriMeshGeom& operator=(const TTriMeshGeom&) {return *this;}

  dGeomID& create() {return _id;}

  void create(dSpaceID space)
    {
      dTriMeshDataID new_tmdata= dGeomTriMeshDataCreate();
      dGeomTriMeshDataBuildSingle(new_tmdata, &vertices_[0], 3*sizeof(float), vertices_.size()/3,
                  &indices_[0], indices_.size(), 3*sizeof(dTriIndex));
      _id= dCreateTriMesh(space, new_tmdata, NULL, NULL, NULL);
    }
  void setBody(dBodyID body)  {dGeomSetBody (_id,body);}

  const std::valarray<float>& getVertices() const {return vertices_;}
  const std::valarray<dTriIndex>& getIndices() const {return indices_;}

  std::valarray<float>& setVertices() {return vertices_;}
  std::valarray<dTriIndex>& setIndices() {return indices_;}

  void setVertices(float *array, int size)
    {
      vertices_.resize(size);
      for(float *itr(&vertices_[0]);size>0;--size,++itr,++array)
        *itr= *array;
    }
  void setIndices(dTriIndex *array, int size)
    {
      indices_.resize(size);
      for(dTriIndex *itr(&indices_[0]);size>0;--size,++itr,++array)
        *itr= *array;
    }

  void getMass(dMass &m, dReal density)
    {
      dMassSetTrimesh(&m, density, _id);
      printf("mass at %f %f %f\n", m.c[0], m.c[1], m.c[2]);
      dGeomSetPosition(_id, -m.c[0], -m.c[1], -m.c[2]);
      dMassTranslate(&m, -m.c[0], -m.c[1], -m.c[2]);
    }

protected:

  std::valarray<float> vertices_;
  std::valarray<dTriIndex> indices_;

};

class TDynRobot
{
public:

  std::vector<TNCBody>& Body() {return body_;}
  dBody& Body(int j) {return body_[j];}

  virtual void Create(dWorldID world, dSpaceID space) = 0;
  void Draw();

  const dReal GetAngleH(int j) const {return joint_h_[j].getAngle();}
  const dReal GetAngVelH(int j) const {return joint_h_[j].getAngleRate();}
  void SetVelH(int j, dReal vel)  {joint_h_[j].setParam(dParamVel,vel);}

protected:
  std::vector<TNCBody> body_;
  std::vector<TNCBox> link_b_;
  std::vector<TNCCapsule> link_ca_;
  std::vector<TNCCylinder> link_cy_;
  std::vector<TNCSphere> link_sp_;
  std::vector<TTriMeshGeom> link_tm_;
  std::vector<TNCBallJoint> joint_b_;
  std::vector<TNCHingeJoint> joint_h_;
  std::vector<TNCHinge2Joint> joint_h2_;
  std::vector<TNCFixedJoint> joint_f_;
};

class TRobot1 : public TDynRobot
{
public:
  double B[3], Size[3];
  bool   HasHinge;
  override void Create(dWorldID world, dSpaceID space);
};

/*override*/void TRobot1::Create(dWorldID world, dSpaceID space)
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();
  link_tm_.clear();

  // dReal side(0.2),mass(1.0);

  body_.resize(5);
  link_b_.resize(5);

  // double outer_size[]= {0.3,0.3,0.4};
  double thickness= BOX_THICKNESS;
  double bottom_z= 0.0;
  double box_sizes[5][3]={
      {thickness, Size[1], Size[2]},
      {thickness, Size[1], Size[2]},
      {Size[0]-2.0*thickness, thickness, Size[2]},
      {Size[0]-2.0*thickness, thickness, Size[2]},
      {Size[0]-2.0*thickness, Size[1]-2.0*thickness, thickness}};
  double box_poss[5][3]={
      {0.5*Size[0]-0.5*thickness, 0.0, 0.5*Size[2]+bottom_z},
      {-0.5*Size[0]+0.5*thickness, 0.0, 0.5*Size[2]+bottom_z},
      {0.0, 0.5*Size[1]-0.5*thickness, 0.5*Size[2]+bottom_z},
      {0.0, -0.5*Size[1]+0.5*thickness, 0.5*Size[2]+bottom_z},
      {0.0, 0.0, 0.5*thickness+bottom_z}};

  for(int i(0); i<5; ++i)
  {
    link_b_[i].create(space, box_sizes[i][0], box_sizes[i][1], box_sizes[i][2]);
    body_[i].create(world);
    body_[i].setPosition(B[0]+box_poss[i][0], B[1]+box_poss[i][1], B[2]+box_poss[i][2]);
    dMass m;
    m.setBox(1.0, box_sizes[i][0], box_sizes[i][1], box_sizes[i][2]);
    body_[i].setMass(&m);
    link_b_[i].setBody(body_[i]);
    link_b_[i].ColorCode= 5;
  }

  // int dir(1);
  // dMatrix3 R;
  // dRSetIdentity (R);  // Z
  // if(dir==1) dRFromAxisAndAngle (R,0.0,1.0,0.0,0.5*M_PI);  // X
  // if(dir==2) dRFromAxisAndAngle (R,1.0,0.0,0.0,0.5*M_PI);  // Y
  // body_[0].setRotation (R);

  joint_f_.resize(4);
  for(int i(0); i<4; ++i)
  {
    joint_f_[i].create(world);
    joint_f_[i].attach(body_[4],body_[i]);
    joint_f_[i].set();
  }

  if(HasHinge)
  {
    joint_h_.resize(1);
    int bid= 3;  // Attaching to this body
    dReal fmax(1000.0);  // NOTE: set zero to control by joint torque
    joint_h_[0].create(world);
    joint_h_[0].attach(body_[bid],0);
    joint_h_[0].setAnchor(B[0]+box_poss[bid][0]+0.5*Size[0], B[1]+box_poss[bid][1], B[2]+box_poss[bid][2]+0.5*Size[2]);
    joint_h_[0].setAxis(0.0,1.0,0.0);
    joint_h_[0].setParam(dParamFMax,fmax);
  }

}

class TBalls1 : public TDynRobot
{
public:
  override void Create(dWorldID world, dSpaceID space);
  std::vector<TNCSphere>& BallsG()  {return link_sp_;}
  const std::vector<TNCSphere>& BallsG() const {return link_sp_;}
  std::vector<TNCBody>& BallsB()  {return body_;}
  const std::vector<TNCBody>& BallsB() const {return body_;}
};

/*override*/void TBalls1::Create(dWorldID world, dSpaceID space)
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();
  link_tm_.clear();

  // dReal side(0.2),mass(1.0);

  int N(100);
  double rad(BALL_RAD);
  double init_z(0.30);

  body_.resize(N);
  link_sp_.resize(N);

  for(int i(0); i<N; ++i)
  {
    double z= init_z + 0.3*rad*double(i);
    double xy_rad= 3.0*rad;
    double th= 0.73*M_PI*double(i);
    link_sp_[i].create(space, rad);
    body_[i].create(world);
    body_[i].setPosition(xy_rad*std::cos(th), xy_rad*std::sin(th), z);
    dMass m;
    m.setSphere(1.0, rad);
    body_[i].setMass(&m);
    link_sp_[i].setBody(body_[i]);
    link_sp_[i].ColorCode= 1;
  }
}

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
  for (std::vector<TNCSphere>::const_iterator itr(link_sp_.begin()),last(link_sp_.end()); itr!=last; ++itr)
  {
    SetColor(itr->ColorCode);
    dsDrawSphere (itr->getPosition(), itr->getRotation(), itr->getRadius());
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
    dsDrawCylinder (pos, rot, VIZ_JOINT_LEN,VIZ_JOINT_RAD);
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

  // TDynRobot& Robot() {return robot_;}

  const dReal& Time() const {return time_;}

  void Create();
  void StepSim(const double &time_step);
  void Draw();

  void ControlCallback(const double &time_step);
  void DrawCallback();

private:
  dWorld world_;
  dSimpleSpace space_;
  dJointGroup contactgroup_;

  TRobot1   source_;
  TRobot1   receiver_;
  TBalls1   balls_;
  // TGeom1    geom_;
  dPlane    plane_;

  dReal time_;
};

void TEnvironment::Create()
{
  contactgroup_.create ();
  world_.setGravity (0,0,-0.5);
  dWorldSetCFM (world_.id(),1e-5);
  plane_.create (space_,0,0,1,0);

  source_.B[0]= 0.0;  source_.B[1]= 0.0;  source_.B[2]= 0.2;
  source_.Size[0]= 0.3;  source_.Size[1]= 0.3;  source_.Size[2]= 0.4;
  source_.HasHinge= true;
  source_.Create(world_,space_);
  receiver_.B[0]= 0.32;  receiver_.B[1]= 0.0;  receiver_.B[2]= 0.0;
  receiver_.Size[0]= 0.3;  receiver_.Size[1]= 0.4;  receiver_.Size[2]= 0.4;
  receiver_.HasHinge= false;
  receiver_.Create(world_,space_);
  balls_.Create(world_,space_);
  // geom_.Create(world_,space_);

  time_= 0.0;
}

void TEnvironment::StepSim(const double &time_step)
{
  ControlCallback(time_step);

  space_.collide (0,&NearCallback);
  world_.step (time_step);
  time_+= time_step;

  contactgroup_.empty();
}

void TEnvironment::Draw()
{
  DrawCallback();

  source_.Draw();
  receiver_.Draw();
  balls_.Draw();
  // geom_.Draw();
}


void TEnvironment::ControlCallback(const double &time_step)
{
  // static double angle = 0;
  // angle += 0.01;
  // robot_.Body().back().addForce (0.1*(std::sin(angle)+1.0), 0.1*(std::sin(angle*1.7)+1.0) ,0.1*(std::sin(angle*0.7)+1.0));
  // robot_.Body().back().addForce (0,0,1.0*(std::sin(angle)+1.0));
  // robot_.Body().front().addForce (0,0,0.01*(std::cos(angle)+1.0));
  dReal Kp(10.0);
  source_.SetVelH(0, Kp*(TargetAngle-source_.GetAngleH(0)));
}

void TEnvironment::DrawCallback()
{
  std::vector<TNCBody> &balls_b(balls_.BallsB());
  std::vector<TNCSphere> &balls_g(balls_.BallsG());
  int num_src(0), num_rcv(0), num_flow(0);
  dReal avr_z_rcv(0.0),speed;
  for(size_t i(0); i<balls_b.size(); ++i)
  {
    const dReal *pos(balls_b[i].getPosition());
    const dReal *vel(balls_b[i].getLinearVel());
    dReal angle= std::atan2(-(pos[0]-0.15),-(pos[2]-0.6));
    speed= std::sqrt(vel[0]*vel[0]+vel[1]*vel[1]+vel[2]*vel[2]);
    if(angle<0.0)  angle+= 2.0*M_PI;
    dReal angle_base= TargetAngle+0.5*M_PI;
    if(pos[0]>0.15 && pos[2]<0.4 && speed<0.1)  {balls_g[i].ColorCode= 0; ++num_rcv; avr_z_rcv+= pos[2]-(BOX_THICKNESS);}
    else if(angle>angle_base)                   {balls_g[i].ColorCode= 2; ++num_flow;}
    else                                        {balls_g[i].ColorCode= 1; ++num_src;}
    if(speed>0.1)  balls_g[i].ColorCode+= 6;
  }
  // std::cerr<<"angle= "<<angle<<"  ";
  if(num_rcv>0)  avr_z_rcv/= double(num_rcv);
  double amount= avr_z_rcv;
  // double amount= 0.0002*double(num_rcv);
  std::cerr<<"#src, #flow, #rcv, amount= "<<num_src<<", "<<num_flow<<", "<<num_rcv<<", "<<amount/*<<", "<<speed*/<<std::endl;
}


static void NearCallback(void *data, dGeomID o1, dGeomID o2)
{
  assert(Env!=NULL);

  // do nothing if the two bodies are connected by a joint
  dBodyID b1 = dGeomGetBody(o1);
  dBodyID b2 = dGeomGetBody(o2);
  if (b1 && b2 && dAreConnected (b1,b2)) return;

  std::valarray<dContact> contact(MAX_CONTACTS);   // up to MAX_CONTACTS contacts per link
  for (int i=0; i<MAX_CONTACTS; i++)
  {
    contact[i].surface.mode = dContactBounce | dContactSoftCFM;
    contact[i].surface.mu = 0.001; // dInfinity;
    contact[i].surface.mu2 = 0.1;
    contact[i].surface.bounce = 0.1;
    contact[i].surface.bounce_vel = 0.01;
    contact[i].surface.soft_cfm = 0.01;
  }
  if (int numc=dCollide (o1,o2,MAX_CONTACTS,&contact[0].geom,sizeof(dContact)))
  {
    for (int i=0; i<numc; i++)
    {
      dJointID c= dJointCreateContact(Env->WorldID(),Env->ContactGroupID(),&contact[i]);
      dJointAttach (c,b1,b2);
    }
  }
}

void Create()
{
  assert(Env!=NULL);
  TargetAngle= 0.0;
  Env->Create();
}

void SimStart()
{
  dAllocateODEDataForThread(dAllocateMaskAll);

  static float xyz[3] = {0.4405,-0.4452,0.8200};
  static float hpr[3] = {123.5000,-35.0000,0.0000};
  dsSetViewpoint (xyz,hpr);
}

void SimLoop (int pause)
{
  assert(Env!=NULL);

  if (!pause)
  {
    Env->StepSim(0.04);
  }

  Env->Draw();
}

void SimKeyevent (int command)
{
  assert(Env!=NULL);

  switch(command)
  {
  case 'r':
  case 'R': Create(); break;
  case 'z': TargetAngle+= 0.01; std::cerr<<"TargetAngle= "<<TargetAngle<<std::endl; break;
  case 'x': TargetAngle-= 0.01; std::cerr<<"TargetAngle= "<<TargetAngle<<std::endl; break;
  }
}

} // end of ode_test

int main (int argc, char **argv)
{
  // setup pointers to drawstuff callback functions
  dsFunctions fn;
  fn.version = DS_VERSION;
  fn.start = &ode_test::SimStart;
  fn.step = &ode_test::SimLoop;
  fn.command = &ode_test::SimKeyevent;
  fn.stop = 0;
  fn.path_to_textures = "textures";

  dInitODE2(0);

  ode_test::TEnvironment env;
  ode_test::Env= &env;
  // env.Create();
  ode_test::Create();

  dsSimulationLoop (argc,argv,500,400,&fn);

  dCloseODE();
  return 0;
}
