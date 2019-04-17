//-------------------------------------------------------------------------------------------
/*! \file    arm7_door_push_node.cpp
    \brief   ODE 7-joint chain simulation with door and pushable object (ROS node).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.28, 2015
*/
//-------------------------------------------------------------------------------------------
#include "ode1/arm7_door_push.h"
#include "ode1/ODEConfig2.h"
#include "ode1/ODESensor2.h"
#include "ode1/ODEControl2.h"
#include "ode1/ODEViz.h"
#include "ode1/ODEVizPrimitive.h"
#include "ode1/ODEGetConfig2.h"
#include "ode1/ODESetConfig2.h"
//-------------------------------------------------------------------------------------------
#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <std_srvs/Empty.h>
//-------------------------------------------------------------------------------------------
namespace trick
{

ros::Publisher *PSensorsPub(NULL), *PKeyEventPub(NULL);

std::vector<ode1::ODEVizPrimitive> VizObjs;

void OnSensingCallback(const ode_x::TSensors2 &sensors)
{
  // Copy sensors to sensor message
  ode1::ODESensor2 sensors_msg;

  sensors_msg.joint_angles= sensors.JointAngles;
  sensors_msg.link_x= sensors.LinkX;
  sensors_msg.forces= sensors.Forces;
  sensors_msg.masses= sensors.Masses;
  sensors_msg.box1_x= sensors.Box1X;
  sensors_msg.chair1_x= sensors.Chair1X;
  sensors_msg.time= sensors.Time;

  PSensorsPub->publish(sensors_msg);
}
//-------------------------------------------------------------------------------------------

void OnDrawCallback()
{
  dReal x[7]={0.0,0.0,0.0, 1.0,0.0,0.0,0.0};
  dReal sides[4]={0.0,0.0,0.0,0.0};
  dMatrix3 R;
  dVector3 p;
  for(std::vector<ode1::ODEVizPrimitive>::const_iterator
      itr(VizObjs.begin()),itr_e(VizObjs.end()); itr!=itr_e; ++itr)
  {
    dsSetColorAlpha(itr->color.r,itr->color.g,itr->color.b,itr->color.a);
    GPoseToX(itr->pose, x);
    dReal q[4]= {x[6],x[3],x[4],x[5]};
    switch(itr->type)
    {
    case ode1::ODEVizPrimitive::LINE:
      p[0]= x[0]+itr->param[0];
      p[1]= x[1]+itr->param[1];
      p[2]= x[2]+itr->param[2];
      dsDrawLine(x, p);
      break;
    case ode1::ODEVizPrimitive::SPHERE:
      dRfromQ(R,q);
      dsDrawSphere(x, R, /*rad=*/itr->param[0]);
      break;
    case ode1::ODEVizPrimitive::CYLINDER:
      dRfromQ(R,q);
      dsDrawCylinder(x, R, /*len=*/itr->param[1], /*rad=*/itr->param[0]);
      break;
    case ode1::ODEVizPrimitive::CUBE:
      dRfromQ(R,q);
      sides[0]= itr->param[0];
      sides[1]= itr->param[1];
      sides[2]= itr->param[2];
      dsDrawBox(x, R, sides);
      break;
    default:
      std::cerr<<"Unknown type:"<<itr->type<<std::endl;
      return;
    }
  }
}
//-------------------------------------------------------------------------------------------

void OnStepCallback(const double &time, const double &time_step)
{
  if(ode_x::Running)
    std::cerr<<"@"<<time<<std::endl;
  else
    usleep(500*1000);
  ros::spinOnce();
  if(!ros::ok())  ode_x::Stop();
}
//-------------------------------------------------------------------------------------------

void OnKeyEventCallback(int command)
{
  std_msgs::Int32 keyevent_msg;
  keyevent_msg.data= command;
  PKeyEventPub->publish(keyevent_msg);
}
//-------------------------------------------------------------------------------------------

void ControlCallback(const ode1::ODEControl2 &msg)
{
  ode_x::TargetAngles = msg.angles ;
  ode_x::TargetGPos   = msg.gpos   ;
  ode_x::TargetVel    = msg.vel    ;
  ode_x::TargetGVel   = msg.gvel   ;
  ode_x::TargetTorque = msg.torque ;
  ode_x::TargetGForce = msg.gforce ;
}
//-------------------------------------------------------------------------------------------

void ODEVizCallback(const ode1::ODEViz &msg)
{
  VizObjs= msg.objects;
}
//-------------------------------------------------------------------------------------------

bool GetConfig(ode1::ODEGetConfig2::Request &req, ode1::ODEGetConfig2::Response &res)
{
  using namespace ode_x;
  ode1::ODEConfig2 &msg(res.config);

  #define COPY(cid)  msg.cid= cid;
  COPY( MaxContacts  )
  COPY( JointNum     )
  COPY( FixedBase    )
  COPY( TotalArmLen  )
  COPY( LinkRad      )
  COPY( FSThick      )
  COPY( FSSize       )
  COPY( BaseLenX     )
  COPY( BaseLenY     )
  COPY( BaseLenZ     )
  COPY( GBaseLenX    )
  COPY( GBaseLenY    )
  COPY( GBaseLenZ    )
  COPY( GripLenY     )
  COPY( GripLenZ     )
  COPY( ObjectMode   )
  COPY( Box1PosX     )
  COPY( Box1PosY     )
  COPY( Box1SizeX    )
  COPY( Box1SizeY    )
  COPY( Box1SizeZ    )
  COPY( Box1Density1 )
  COPY( Box1Density2 )
  COPY( Chair1PosX         )
  COPY( Chair1PosY         )
  COPY( Chair1BaseRad      )
  COPY( Chair1BaseLen      )
  COPY( Chair1Caster1Rad   )
  COPY( Chair1Caster2Rad   )
  COPY( Chair1Caster3Rad   )
  COPY( Chair1Caster4Rad   )
  COPY( Chair1CasterDX     )
  COPY( Chair1CasterDY     )
  COPY( Chair1Seat1Density )
  COPY( Chair1Seat1DX      )
  COPY( Chair1Seat1DY      )
  COPY( Chair1Seat1SizeX   )
  COPY( Chair1Seat1SizeY   )
  COPY( Chair1Seat1SizeZ   )
  COPY( Chair1Seat2Density )
  COPY( Chair1Seat2DX      )
  COPY( Chair1Seat2DY      )
  COPY( Chair1Seat2SizeX   )
  COPY( Chair1Seat2SizeY   )
  COPY( Chair1Seat2SizeZ   )
  COPY( Chair1Damping      )
  COPY( TimeStep     )
  COPY( Gravity      )
  COPY( EnableKeyEvent )
  COPY( HingeFMax    )
  COPY( SliderFMax   )
  COPY( ControlMode  )
  #undef COPY

  return true;
}
//-------------------------------------------------------------------------------------------

bool ResetSim2(ode1::ODESetConfig2::Request &req, ode1::ODESetConfig2::Response &res)
{
  using namespace ode_x;
  std::cerr<<"Resetting simulator..."<<std::endl;
  const ode1::ODEConfig2 &msg(req.config);

  #define COPY(cid)  cid= msg.cid;
  COPY( MaxContacts  )
  COPY( JointNum     )
  COPY( FixedBase    )
  COPY( TotalArmLen  )
  COPY( LinkRad      )
  COPY( FSThick      )
  COPY( FSSize       )
  COPY( BaseLenX     )
  COPY( BaseLenY     )
  COPY( BaseLenZ     )
  COPY( GBaseLenX    )
  COPY( GBaseLenY    )
  COPY( GBaseLenZ    )
  COPY( GripLenY     )
  COPY( GripLenZ     )
  COPY( ObjectMode   )
  COPY( Box1PosX     )
  COPY( Box1PosY     )
  COPY( Box1SizeX    )
  COPY( Box1SizeY    )
  COPY( Box1SizeZ    )
  COPY( Box1Density1 )
  COPY( Box1Density2 )
  COPY( Chair1PosX         )
  COPY( Chair1PosY         )
  COPY( Chair1BaseRad      )
  COPY( Chair1BaseLen      )
  COPY( Chair1Caster1Rad   )
  COPY( Chair1Caster2Rad   )
  COPY( Chair1Caster3Rad   )
  COPY( Chair1Caster4Rad   )
  COPY( Chair1CasterDX     )
  COPY( Chair1CasterDY     )
  COPY( Chair1Seat1Density )
  COPY( Chair1Seat1DX      )
  COPY( Chair1Seat1DY      )
  COPY( Chair1Seat1SizeX   )
  COPY( Chair1Seat1SizeY   )
  COPY( Chair1Seat1SizeZ   )
  COPY( Chair1Seat2Density )
  COPY( Chair1Seat2DX      )
  COPY( Chair1Seat2DY      )
  COPY( Chair1Seat2SizeX   )
  COPY( Chair1Seat2SizeY   )
  COPY( Chair1Seat2SizeZ   )
  COPY( Chair1Damping      )
  COPY( TimeStep     )
  COPY( Gravity      )
  COPY( EnableKeyEvent )
  COPY( HingeFMax    )
  COPY( SliderFMax   )
  COPY( ControlMode  )
  #undef COPY

  ode_x::Reset();
  std::cerr<<"done."<<std::endl;
  return true;
}
//-------------------------------------------------------------------------------------------

bool ResetSim(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
  std::cerr<<"Resetting simulator..."<<std::endl;
  ode_x::Reset();
  std::cerr<<"done."<<std::endl;
  return true;
}
//-------------------------------------------------------------------------------------------

bool Pause(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
  std::cerr<<"Simulator paused..."<<std::endl;
  ode_x::Running= false;
  return true;
}
//-------------------------------------------------------------------------------------------

bool Resume(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
  std::cerr<<"Simulator resumed..."<<std::endl;
  ode_x::Running= true;
  return true;
}
//-------------------------------------------------------------------------------------------

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  ros::init(argc, argv, "chain_sim");
  ros::NodeHandle node("~");

  std::string texture_path;
  int winx(500), winy(400);
  node.param("texture_path",texture_path,std::string("config/textures"));
  node.param("winx",winx,winx);
  node.param("winy",winy,winy);

  ros::Publisher sensors_pub= node.advertise<ode1::ODESensor2>("sensors", 1);
  PSensorsPub= &sensors_pub;
  ros::Publisher keyevent_pub= node.advertise<std_msgs::Int32>("keyevent", 1);
  PKeyEventPub= &keyevent_pub;

  ros::Subscriber sub_control= node.subscribe("control", 1, &ControlCallback);
  ros::Subscriber sub_viz= node.subscribe("viz", 1, &ODEVizCallback);
  ros::ServiceServer srv_get_config= node.advertiseService("get_config", &GetConfig);
  ros::ServiceServer srv_reset= node.advertiseService("reset", &ResetSim);
  ros::ServiceServer srv_reset2= node.advertiseService("reset2", &ResetSim2);
  ros::ServiceServer srv_pause= node.advertiseService("pause", &Pause);
  ros::ServiceServer srv_resume= node.advertiseService("resume", &Resume);

  ode_x::SensingCallback= &OnSensingCallback;
  ode_x::DrawCallback= &OnDrawCallback;
  ode_x::StepCallback= &OnStepCallback;
  ode_x::KeyEventCallback= &OnKeyEventCallback;
  ode_x::Run(argc, argv, texture_path.c_str(), winx, winy);

  return 0;
}
//-------------------------------------------------------------------------------------------
