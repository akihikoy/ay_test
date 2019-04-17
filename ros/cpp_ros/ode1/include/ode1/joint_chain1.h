//-------------------------------------------------------------------------------------------
/*! \file    joint_chain1.h
    \brief   ODE N-joint chain simulation 1.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.05, 2015
*/
//-------------------------------------------------------------------------------------------
#ifndef joint_chain1_h
#define joint_chain1_h
//-------------------------------------------------------------------------------------------
#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#ifdef dDOUBLE
#define dsDrawBox dsDrawBoxD
#define dsDrawSphere dsDrawSphereD
#define dsDrawCylinder dsDrawCylinderD
#define dsDrawCapsule dsDrawCapsuleD
#define dsDrawTriangle dsDrawTriangleD
#define dsDrawLine dsDrawLineD
#endif
//-------------------------------------------------------------------------------------------
#include <vector>
#include <valarray>
//-------------------------------------------------------------------------------------------
#ifndef override
#define override
#endif
//-------------------------------------------------------------------------------------------
namespace trick
{
//-------------------------------------------------------------------------------------------

// Convert geometry_msgs/Pose to x; usually, t_pose==geometry_msgs::Pose
template <typename t_pose, typename t_value>
inline void GPoseToX(const t_pose &pose, t_value x[7])
{
  x[0]= pose.position.x;
  x[1]= pose.position.y;
  x[2]= pose.position.z;
  x[3]= pose.orientation.x;
  x[4]= pose.orientation.y;
  x[5]= pose.orientation.z;
  x[6]= pose.orientation.w;
}
//-------------------------------------------------------------------------------------------


namespace ode_x
{
//-------------------------------------------------------------------------------------------

struct TSensors1;

extern int MaxContacts;  // maximum number of contact points per body
extern double VizJointLen;
extern double VizJointRad;
extern int    JointNum;
extern double TotalLen;
extern double LinkRad;
extern std::vector<double> TargetAngles;
extern double TimeStep;
extern double Gravity;
extern bool Running;
extern void (*SensingCallback)(const TSensors1 &sensors);
extern void (*DrawCallback)(void);
extern void (*StepCallback)(const double &time, const double &time_step);
//-------------------------------------------------------------------------------------------

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
DEF_NC(SliderJoint,3)
DEF_NC(FixedJoint,4)
DEF_NC(Box,0)
DEF_NC(Capsule,1)
DEF_NC(Cylinder,2)
DEF_NC(Sphere,3)
#undef DEF_NC
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TTriMeshGeom : public dGeom
//===========================================================================================
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
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TDynRobot
//===========================================================================================
{
public:

  std::vector<TNCBody>& Body() {return body_;}
  dBody& Body(int j) {return body_[j];}

  virtual void Create(dWorldID world, dSpaceID space) = 0;
  void Draw();

  const dReal GetAngleH(int j) const {return joint_h_[j].getAngle();}
  const dReal GetAngVelH(int j) const {return joint_h_[j].getAngleRate();}
  void SetVelH(int j, dReal vel)  {joint_h_[j].setParam(dParamVel,vel);}

  const dReal GetPosS(int j) const {return joint_s_[j].getPosition();}
  const dReal GetVelS(int j) const {return joint_s_[j].getPositionRate();}
  void SetVelS(int j, dReal vel)  {joint_s_[j].setParam(dParamVel,vel);}

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
  std::vector<TNCSliderJoint> joint_s_;
  std::vector<TNCFixedJoint> joint_f_;
};
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TJointChain1 : public TDynRobot
//===========================================================================================
{
public:
  override void Create(dWorldID world, dSpaceID space);
};
//-------------------------------------------------------------------------------------------

struct TSensors1
{
  std::vector<double> JointAngles;
  std::vector<double> LinkX;  // Poses (x,y,z,qx,qy,qz,qw) of links; [7]*(JointNum+1)

  double Time;  // Simulation time

  void Clear()
    {
      JointAngles.clear();
      LinkX.clear();
      SetZeros(-1);
    }

  void SetZeros(int num_joints)
    {
      if(num_joints>=0)
      {
        JointAngles.resize(num_joints);
        LinkX.resize(7*(num_joints+1));
      }
      for(std::vector<double>::iterator itr(JointAngles.begin()),itr_end(JointAngles.end());
          itr!=itr_end; ++itr)
        *itr= 0;
      for(std::vector<double>::iterator itr(LinkX.begin()),itr_end(LinkX.end());
          itr!=itr_end; ++itr)
        *itr= 0;
      Time= 0.0;
    }

  // Reset values for each new physics computation step.
  // Only collision flags are reset. Other values are kept.
  void ResetForStep()
    {
    }
};
//-------------------------------------------------------------------------------------------

//===========================================================================================
class TEnvironment
//===========================================================================================
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
  void EDrawCallback();
  /* Called when b1 and b2 are colliding.
      Return whether we ignore this collision (true: ignore collision). */
  bool CollisionCallback(dBodyID &b1, dBodyID &b2, std::valarray<dContact> &contact);

private:
  dWorld world_;
  dSimpleSpace space_;
  dJointGroup contactgroup_;

  TJointChain1     chain_;
  // TGeom1    geom_;
  dPlane    plane_;

  dReal time_;

  TSensors1  sensors_;
};
//-------------------------------------------------------------------------------------------


void Create();
void Reset();
void Run(int argc, char **argv, const char *texture_path="textures", int winx=500, int winy=400);
void Stop();
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
}  // end of ode_x
//-------------------------------------------------------------------------------------------
}  // end of trick
//-------------------------------------------------------------------------------------------
#endif // joint_chain1_h
//-------------------------------------------------------------------------------------------
