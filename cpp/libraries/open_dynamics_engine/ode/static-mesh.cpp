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
DEF_NC(BallJoint)
DEF_NC(HingeJoint)
DEF_NC(Hinge2Joint)
DEF_NC(FixedJoint)
DEF_NC(Box)
DEF_NC(Capsule)
DEF_NC(Cylinder)
DEF_NC(Sphere)
#undef DEF_NC

class TTriMeshGeom : public dGeom
{
public:
  TTriMeshGeom() : dGeom() {}

  TTriMeshGeom(const TTriMeshGeom&) : dGeom() {}
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
  void SetVelH(int j, dReal vel)  {joint_h_[j].setParam(dParamVel,vel);}
  void GetHandPos(dReal *pos);

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

  dReal side(0.2),mass(1.0);

  body_.resize(10);
  link_b_.resize(10);
  for (int i=0; i<10; i++)
  {
    body_[i].create (world);
    // dReal k = i*side;
    // body_[i].setPosition (k,k,k+0.4);
    dReal k = 1.5*i*side;
    body_[i].setPosition (0,0,k+1.0);
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
    // dReal k = (i+0.5)*side;
    // joint_b_[i].setAnchor (k,k,k+0.4);
    dReal k = 1.5*(i+0.5)*side;
    joint_b_[i].setAnchor (0,0,k+1.0);
  }
}

class TGeom1 : public TDynRobot
{
public:
  override void Create(dWorldID world, dSpaceID space);
};

/*override*/void TGeom1::Create(dWorldID world, dSpaceID space)
{
  body_.clear();
  link_b_.clear();
  link_ca_.clear();
  link_cy_.clear();
  link_sp_.clear();
  link_tm_.clear();

  link_tm_.resize(1);

  {
    const dReal size(5.0);
    const int cols(20);  // # of x-points of grid
    const int rows(20);  // # of y-points of grid
    const int vertex_count = rows*cols;
    const int index_count = (rows-1)*2*(cols-1) * 3;

    float vertices[vertex_count * 3];
    dTriIndex indices[index_count];

    double cstep(size/dReal(cols-1));
    double rstep(size/dReal(rows-1));
    for(int c(0);c<cols;++c)
    {
      for(int r(0);r<rows;++r)
      {
        vertices[3*(c*rows+r)+0]= cstep*dReal(c);// + cstep*0.4*(dRandReal()-0.5);
        vertices[3*(c*rows+r)+1]= rstep*dReal(r);// + rstep*0.4*(dRandReal()-0.5);
        vertices[3*(c*rows+r)+2]= 0.4*dRandReal();
      }
    }
    for(int c(0);c<cols-1;++c)
    {
      for(int r(0);r<rows-1;++r)
      {
        indices[3*(2*(c*(rows-1)+r)+0)+0]= c*rows+r;
        indices[3*(2*(c*(rows-1)+r)+0)+1]= (c+1)*rows+(r+1);
        indices[3*(2*(c*(rows-1)+r)+0)+2]= c*rows+(r+1);

        indices[3*(2*(c*(rows-1)+r)+1)+0]= c*rows+r;
        indices[3*(2*(c*(rows-1)+r)+1)+1]= (c+1)*rows+r;
        indices[3*(2*(c*(rows-1)+r)+1)+2]= (c+1)*rows+(r+1);
      }
    }
for(int i(0);i<vertex_count;++i)
  std::cerr<<vertices[3*i]<<" "<<vertices[3*i+1]<<" "<<vertices[3*i+2]<<std::endl;
for(int i(0);i<index_count/3;++i)
  std::cerr<<indices[3*i]<<" "<<indices[3*i+1]<<" "<<indices[3*i+2]<<std::endl;

    link_tm_[0].setVertices(vertices,vertex_count*3);
    link_tm_[0].setIndices(indices,index_count);
    link_tm_[0].create(space);
    link_tm_[0].setPosition (-0.5*size,-0.5*size,0);
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
  dsSetColorAlpha (1.0, 0.0, 0.5, 0.6);
  for (std::vector<TTriMeshGeom>::const_iterator itr(link_tm_.begin()),last(link_tm_.end()); itr!=last; ++itr)
  {
    // const dReal *pos = itr->getPosition();
    // const dReal *rot = itr->getRotation();
    const dVector3 pos={0,0,0,0};
    const dMatrix3 rot={1,0,0,0 ,0,1,0,0, 0,0,1,0};

    for (int i(dGeomTriMeshGetTriangleCount(*itr)); i>0; --i)
    {
      // std::cerr<<"i:"<<i<<std::endl;
      dVector3 v[3];
      dGeomTriMeshGetTriangle(*itr, i-1, &v[0], &v[1], &v[2]);
      // std::cerr<<v[0][0]<<" "<<v[0][1]<<" "<<v[0][2]<<std::endl;
      // std::cerr<<v[1][0]<<" "<<v[1][1]<<" "<<v[1][2]<<std::endl;
      // std::cerr<<v[2][0]<<" "<<v[2][1]<<" "<<v[2][2]<<std::endl;
      dsDrawTriangle(pos, rot, v[0], v[1], v[2], 1);
    }
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

  TRobot1   robot_;
  TGeom1    geom_;
  dPlane    plane_;
};

void TEnvironment::Create()
{
  contactgroup_.create (0);
  world_.setGravity (0,0,-0.5);
  dWorldSetCFM (world_.id(),1e-5);
  plane_.create (space_,0,0,1,0);

  robot_.Create(world_,space_);
  geom_.Create(world_,space_);
}

void TEnvironment::StepSim(const double &time_step)
{
  static double angle = 0;
  angle += 0.01;
  // robot_.Body().back().addForce (0.1*(std::sin(angle)+1.0), 0.1*(std::sin(angle*1.7)+1.0) ,0.1*(std::sin(angle*0.7)+1.0));
  robot_.Body().back().addForce (0,0,1.0*(std::sin(angle)+1.0));
  robot_.Body().front().addForce (0,0,0.01*(std::cos(angle)+1.0));

  space_.collide (0,&NearCallback);
  world_.step (time_step);

  contactgroup_.empty();
}

void TEnvironment::Draw()
{
  robot_.Draw();
  geom_.Draw();
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
