//-------------------------------------------------------------------------------------------
/*! \file    manip01_ctrl.h
    \brief   Controller for manipulator01
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Apr.15, 2014
*/
//-------------------------------------------------------------------------------------------
#ifndef manip01_ctrl_h
#define manip01_ctrl_h
//-------------------------------------------------------------------------------------------
#include <lora/robot_model.h>
#include "kinematics.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------

/*!\brief generate kinematics::TKinematicRobot from xode::TRobotParameters */
void GenerateKinematicRobotModel(kinematics::TKinematicRobot &robot, const xode::TRobotParameters &src);

// \FIXME following should be in a source file

inline void CopyPos(kinematics::DVector3 dest, const std::vector<TReal> &src)
{
  xode::VectorToPosition(src,dest);
}

inline void CopyRot(kinematics::DMatrix3 dest, const std::vector<TReal> &src)
{
  if(src.size()>0)  xode::VectorToRotation(src,dest);
  else  dRSetIdentity(dest);
}
//-------------------------------------------------------------------------------------------

void AssignConnections(kinematics::TKinematicRobot &robot, int b_idx, int j_parent)
{
  for(int j(0); j<robot.JointNum(); ++j)
  {
    if(j==j_parent)  continue;
    if(robot.Joint(j).B1==b_idx || robot.Joint(j).B2==b_idx)
    {
      robot.AddConnectedJointToLink(b_idx, j);
      int b2_idx= robot.Joint(j).B1;
      if(b2_idx==b_idx)  b2_idx= robot.Joint(j).B2;
      AssignConnections(robot, b2_idx, j);
    }
  }
}
//-------------------------------------------------------------------------------------------

/*!\brief generate kinematics::TKinematicRobot from xode::TRobotParameters */
void GenerateKinematicRobotModel(kinematics::TKinematicRobot &robot, const xode::TRobotParameters &src)
{
  robot.Clear();

  for(xode::TNamedParam<xode::TLinkParameters>::P::const_iterator litr(src.Links.begin()); litr!=src.Links.end(); ++litr)
  {
    // litr->first : name
    // litr->second  : TLinkParameters
    kinematics::TKinematicLink new_link;
    new_link.LinkID= litr->first;
    CopyRot(new_link.BaseRot, litr->second.Rotation);
    CopyPos(new_link.BasePos, litr->second.Position);
    CopyRot(new_link.R, litr->second.Rotation);
    CopyPos(new_link.p, litr->second.Position);
    robot.AddLink (new_link);
  }

  for(xode::TNamedParam<xode::TJointParameters>::P::const_iterator jitr(src.Joints.begin()); jitr!=src.Joints.end(); ++jitr)
  {
    // jitr->first : name
    // jitr->second : TJointParameters
    if(jitr->second.Body1=="-" || jitr->second.Body2=="-")  continue;

    kinematics::TKinematicJoint new_joint;
    new_joint.JointID= jitr->first;
    new_joint.B1= robot.LinkIDToIndex(jitr->second.Body1);
    new_joint.B2= robot.LinkIDToIndex(jitr->second.Body2);

    if(jitr->second.Type==xode::jtHinge)
    {
      new_joint.Type= kinematics::jtHinge;
      CopyPos(new_joint.BaseAnchor, jitr->second.Anchor);
      CopyPos(new_joint.BaseAxis1, jitr->second.Axis1);
      new_joint.Lo1= jitr->second.AParam1.LoStop;
      new_joint.Hi1= jitr->second.AParam1.HiStop;
      new_joint.Offset1= 0.0;

      CopyPos(new_joint.Anchor, jitr->second.Anchor);
      CopyPos(new_joint.Axis1, jitr->second.Axis1);
      new_joint.q1= 0.0;
    }
    else if(jitr->second.Type==xode::jtSlider)
    {
      new_joint.Type= kinematics::jtSlider;
      CopyPos(new_joint.BaseAxis1, jitr->second.Axis1);
      new_joint.Lo1= jitr->second.AParam1.LoStop;
      new_joint.Hi1= jitr->second.AParam1.HiStop;
      new_joint.Offset1= 0.0;

      CopyPos(new_joint.Axis1, jitr->second.Axis1);
      new_joint.q1= 0.0;
    }
    else if(jitr->second.Type==xode::jtUniversal)
    {
      new_joint.Type= kinematics::jtUniversal;
      CopyPos(new_joint.BaseAnchor, jitr->second.Anchor);
      CopyPos(new_joint.BaseAxis1, jitr->second.Axis1);
      CopyPos(new_joint.BaseAxis2, jitr->second.Axis2);
      new_joint.Lo1= jitr->second.AParam1.LoStop;
      new_joint.Hi1= jitr->second.AParam1.HiStop;
      new_joint.Lo2= jitr->second.AParam2.LoStop;
      new_joint.Hi2= jitr->second.AParam2.HiStop;
      new_joint.Offset1= 0.0;
      new_joint.Offset2= 0.0;

      CopyPos(new_joint.Anchor, jitr->second.Anchor);
      CopyPos(new_joint.Axis1, jitr->second.Axis1);
      CopyPos(new_joint.Axis2, jitr->second.Axis2);
      new_joint.q1= 0.0;
      new_joint.q2= 0.0;
    }
    // else if(jitr->second.Type==xode::jtHinge2)
    // {
    // }
    // else if(jitr->second.Type==xode::jtFixed)
    // {
      // new_joint.Type= kinematics::jtFixed;
    // }
    else
    {
      LWARNING("in GenerateKinematicRobotModel, not implemented joint type ("<<jitr->second.Type<<")");
    }

    robot.AddJoint(new_joint);
  }

  AssignConnections(robot, robot.LinkIDToIndex(src.RootLink), -1);
  robot.SetBaseLinkIdx(robot.LinkIDToIndex(src.RootLink));

  robot.SetChangeableLinks();
  robot.ClearEndEffectors();
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
}  // end of loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // manip01_ctrl_h
//-------------------------------------------------------------------------------------------
