//-------------------------------------------------------------------------------------------
/*! \file    joint_chain1_node.cpp
    \brief   ODE N-joint chain simulation 1.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.05, 2015
*/
//-------------------------------------------------------------------------------------------
#include "ode1/joint_chain1.h"
#include "ode1/ODEConfig.h"
#include "ode1/ODESensor.h"
#include "ode1/ODEViz.h"
#include "ode1/ODEVizPrimitive.h"
#include "ode1/ODEGetConfig.h"
#include "ode1/ODEReset2.h"
//-------------------------------------------------------------------------------------------
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_srvs/Empty.h>
//-------------------------------------------------------------------------------------------
namespace trick
{

ros::Publisher *PSensorsPub(NULL);

std::vector<ode1::ODEVizPrimitive> VizObjs;

void OnSensingCallback(const ode_x::TSensors1 &sensors)
{
  // Copy sensors to sensor message
  ode1::ODESensor sensors_msg;

  sensors_msg.joint_angles= sensors.JointAngles;
  sensors_msg.link_x= sensors.LinkX;
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

void ThetaCallback(const std_msgs::Float64MultiArray &msg)
{
  if(msg.data.size()>ode_x::JointNum)
  {
    std::cerr<<"ThetaCallback: number of angles should be less than "<<ode_x::JointNum
        <<" but given "<<msg.data.size()<<std::endl;
    return;
  }
  for(int j(0),j_end(msg.data.size()); j<j_end; ++j)
    ode_x::TargetAngles[j]= msg.data[j];
}
//-------------------------------------------------------------------------------------------

void ODEVizCallback(const ode1::ODEViz &msg)
{
  VizObjs= msg.objects;
}
//-------------------------------------------------------------------------------------------

bool GetConfig(ode1::ODEGetConfig::Request &req, ode1::ODEGetConfig::Response &res)
{
  using namespace ode_x;
  ode1::ODEConfig &msg(res.config);

  msg.MaxContacts       = MaxContacts      ;
  msg.JointNum          = JointNum         ;
  msg.TotalLen          = TotalLen         ;
  msg.LinkRad           = LinkRad          ;
  msg.TimeStep          = TimeStep         ;
  msg.Gravity           = Gravity          ;

  return true;
}
//-------------------------------------------------------------------------------------------

bool ResetSim2(ode1::ODEReset2::Request &req, ode1::ODEReset2::Response &res)
{
  using namespace ode_x;
  std::cerr<<"Resetting simulator..."<<std::endl;
  const ode1::ODEConfig &msg(req.config);
  MaxContacts       = msg.MaxContacts   ;
  JointNum          = msg.JointNum      ;
  TotalLen          = msg.TotalLen      ;
  LinkRad           = msg.LinkRad       ;
  TimeStep          = msg.TimeStep      ;
  Gravity           = msg.Gravity       ;
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

  ros::Publisher sensors_pub= node.advertise<ode1::ODESensor>("sensors", 1);
  PSensorsPub= &sensors_pub;

  ros::Subscriber sub_theta= node.subscribe("theta", 1, &ThetaCallback);
  ros::Subscriber sub_viz= node.subscribe("viz", 1, &ODEVizCallback);
  ros::ServiceServer srv_get_config= node.advertiseService("get_config", &GetConfig);
  ros::ServiceServer srv_reset= node.advertiseService("reset", &ResetSim);
  ros::ServiceServer srv_reset2= node.advertiseService("reset2", &ResetSim2);
  ros::ServiceServer srv_pause= node.advertiseService("pause", &Pause);
  ros::ServiceServer srv_resume= node.advertiseService("resume", &Resume);

  ode_x::SensingCallback= &OnSensingCallback;
  ode_x::DrawCallback= &OnDrawCallback;
  ode_x::StepCallback= &OnStepCallback;
  ode_x::Run(argc, argv, texture_path.c_str(), winx, winy);

  return 0;
}
//-------------------------------------------------------------------------------------------
