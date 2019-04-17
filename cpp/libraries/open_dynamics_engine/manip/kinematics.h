//-------------------------------------------------------------------------------------------
/*! \file    kinematics.h
    \brief   Kinematics solver
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \date    Nov.20, 2009
*/
//-------------------------------------------------------------------------------------------
#ifndef loco_rabbits_kinematics_h
#define loco_rabbits_kinematics_h
//-------------------------------------------------------------------------------------------
#include <list>
#include <map>
#include <vector>
#include <lora/octave.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
namespace kinematics
{
//-------------------------------------------------------------------------------------------

typedef ColumnVector TRealVector;
typedef Matrix       TRealMatrix;

typedef double DReal;
typedef DReal  DVector3[4];
typedef DReal  DMatrix3[4*3];

typedef std::string TJointID;
typedef std::string TLinkID;

struct TKinematicLink;

enum TJointType {
  jtBall      =0,
  jtHinge     ,
  jtSlider    ,
  jtUniversal ,
  jtHinge2    ,
  jtFixed     };
//-------------------------------------------------------------------------------------------

/*!\brief joint object */
struct TKinematicJoint
{
  TJointID JointID;  //!< a pointer to the corresponding joint of a robot model (e.g. ODE::dJointID)
  std::vector<bool>  ChangeableLinks;

  int  B1,  B2;   // index in links_
  TJointType Type;
  DVector3  BaseAnchor, BaseAxis1, BaseAxis2;  //!< base joint anchor, axis1, axis2 in world frame
  DReal     Lo1, Hi1, Lo2, Hi2;   //!< joint(or position)-range
  DReal     Offset1, Offset2;  //!< offset of angle (or position)

  DVector3  Anchor, Axis1, Axis2;  //!< current joint anchor, axis1, axis2 in world frame
  DReal     q1, q2;  //!< current angle (or position)
};
//-------------------------------------------------------------------------------------------

// typedef std::list<TKinematicJoint*>  TJointList;
typedef std::list<int>  TJointList;

/*!\brief link object */
struct TKinematicLink
{
  TLinkID LinkID;  //!< a pointer to the corresponding link of a robot model (e.g. ODE::dBodyID)

  TJointList  Connected;  //!< list of connected joints to this link
  DMatrix3  BaseRot;  //!< base rotation in world frame
  DVector3  BasePos;  //!< base position in world frame

  DMatrix3  R;  //!< current rotation in world frame
  DVector3  p;  //!< current position in world frame

  DMatrix3  TargetRot;  //!< target rotation in world frame (used when this link is an end-effector)
  DVector3  TargetPos;  //!< target position in world frame  (used when this link is an end-effector)
};
//-------------------------------------------------------------------------------------------


/*!\brief link structure
  \warning current implementation supports TREE STRUCTURE ONLY.
  usage:\code
    generateStructure (body0);
    addEndEffectorPos (body11);
    addEndEffectorPos (body12);
    addEndEffectorRot (body11);

    addMovableJoint (joint1);
    addMovableJoint (joint2);
    addMovableJoint (joint4);

    setEndEffectorTargetPos (body11, x11);
    setEndEffectorTargetPos (body12, x12);
    setEndEffectorTargetRot (body11, R11);
    execInverseKinematics();
  \endcode
 */
class TKinematicRobot
{
public:

  enum TInverseType {
    itPseudoInverse = 0 /*! precise */,
    itTranspose         /*! fast */
  };

  TKinematicRobot() : base_link_idx_(-1) {}

  //!\brief clear the model
  void Clear (void);

  const TKinematicLink& Link(int j) const {return links_[j];}
  const TKinematicJoint& Joint(int j) const {return joints_[j];}

  int LinkNum() const {return links_.size();}
  int JointNum() const {return joints_.size();}

  int LinkIDToIndex(const TLinkID &linkid)  {return linkid_to_idx_[linkid];}
  int JointIDToIndex(const TJointID &jointid)  {return jointid_to_idx_[jointid];}

  //!\brief add movable joint that is used to solve inverse kinematics
  void AddMovableJoint (const TJointID &jointid);
  //!\brief clear movable joints
  void ClearMovableJoints ();

  int MovableJointNum() const {return anglevec_idx_to_q_ptr_.size();}
  DReal& MovableJointAngle(int idx)  {return *anglevec_idx_to_q_ptr_[idx];}
  const DReal& MovableJointAngle(int idx) const {return *anglevec_idx_to_q_ptr_[idx];}

  //!\brief add an end effector (position)
  void AddEndEffectorPos (const TLinkID &linkid);
  //!\brief add an end effector (rotation)
  void AddEndEffectorRot (const TLinkID &linkid);
  //!\brief clear every end effectors
  void ClearEndEffectors ();

  //!\brief set the target position of the link
  void SetEndEffectorTargetPos (const TLinkID &linkid, DVector3 target_pos);
  //!\brief set the target rotation of the link
  void SetEndEffectorTargetRot (const TLinkID &linkid, DMatrix3 target_rot);
  //!\brief execute inverse kinematics (from current pose)
  DReal ExecInverseKinematics (const DReal &tol=1.0e-6, int MaxItr=1000, const DReal &step_size=0.2, TInverseType inv_type=itPseudoInverse);

  //!\brief return joint type
  TJointType GetJointType (const TJointID &jointid) const
    {
      std::map <TJointID, int>::const_iterator ij= jointid_to_idx_.find(jointid);
      if (ij==jointid_to_idx_.end())  {LERROR("fatal!");}
      return joints_[ij->second].Type;
    }

  //!\brief return joint angle1 resulted from the inverse kinematics
  const DReal& GetJointAngle1 (const TJointID &jointid) const
    {
      std::map <TJointID, int>::const_iterator ij= jointid_to_idx_.find(jointid);
      if (ij==jointid_to_idx_.end())  {LERROR("fatal!");}
      return joints_[ij->second].q1;
    }
  //!\brief return joint angle2 (for jtUniversal) resulted from the inverse kinematics
  const DReal& GetJointAngle2 (const TJointID &jointid) const
    {
      std::map <TJointID, int>::const_iterator ij= jointid_to_idx_.find(jointid);
      if (ij==jointid_to_idx_.end())  {LERROR("fatal!");}
      return joints_[ij->second].q2;
    }
  //!\brief return joint position (for jtSlider) resulted from the inverse kinematics
  const DReal& GetJointPosition (const TJointID &jointid) const
    {
      std::map <TJointID, int>::const_iterator ij= jointid_to_idx_.find(jointid);
      if (ij==jointid_to_idx_.end())  {LERROR("fatal!");}
      return joints_[ij->second].q1;
    }

  //-----------------------------------------------
  // methods to construct the link structure

  int NewLinkIndex() const {return links_.size();}

  int AddLink (const TKinematicLink &link);
  int AddJoint (const TKinematicJoint &joint);
  void AddConnectedJointToLink (int link_idx, int joint_idx);

  void SetBaseLinkIdx (int link_idx)  {base_link_idx_= link_idx;}

  //!\brief set ChangeableLinks
  void SetChangeableLinks (void);

private:
  typedef std::vector<TKinematicLink>   TKinematicLinkSet;
  typedef std::vector<TKinematicJoint>  TKinematicJointSet;

  TKinematicLinkSet   links_;
  TKinematicJointSet  joints_;

  int  base_link_idx_;

  // mapper between the internal model and the ODE model
  std::map <TLinkID, int>   linkid_to_idx_;    //!< get link index in links_ from TLinkID
  std::map <TJointID, int>  jointid_to_idx_;   //!< get joint index in joints_ from TJointID

  // for forward and inverse kinematics
  typedef std::map<TKinematicJoint*,int>  TMovableJointMap;
  std::vector<DReal*>         anglevec_idx_to_q_ptr_;   //!< get a pointer to q1/q2 from index in angle vector
  TMovableJointMap            movable_jointptr_to_anglevec_idx_;  //!< get index of q1 in angle vector from joint pointer

  std::vector<int>            end_effector_pos_list_;  //!< list of positional end-effector link (in links_)
  std::vector<int>            end_effector_rot_list_;  //!< list of rotational end-effector link (in links_)
  bool   end_effector_list_is_changed_;

  // temporary variables:
  TRealVector dq_, err_vector_;
  TRealMatrix J_;
  // temporary variables:
  TRealMatrix Rerr_, target_link_R_;
  TRealVector werr_;

  /*! \brief rotate all succeeding links connected to ib in rotation rot around pos */
  void rot_succeeding_links_ (TKinematicLink *ib, TKinematicJoint *parent, const DVector3 pos, const DMatrix3 rot);
  /*! \brief move all succeeding links connected to ib in distance d */
  void move_succeeding_links_ (TKinematicLink *ib, TKinematicJoint *parent, const DVector3 d);

  //!\brief reset current p,R,Anchor,Axis1,Axis2 to base line.
  void reset_to_base_ ();
  //!\brief set joint angles (or positions) which are set to q1/q2 (robot is rotated; forward kinematics)
  void set_joint_angles_ (TKinematicLink *ib, TKinematicJoint *parent);
  /*!\brief add dq to joint angles (robot is rotated; forward kinematics)
      \param [in]dq  joint angles (movable_joint_ only) */
  void add_to_joint_angles_ (TKinematicLink *ib, TKinematicJoint *parent, const double *dq);
  //!\brief calculate Jacobian at the current pose and asign it into J
  void get_current_jacobian_ (TRealMatrix &J);

  //!\brief calculate error vector
  void get_error_vector_ (double *err_vector);

  //!\brief set ChangeableLinks
  void set_changeable_links_ (TKinematicLink *ib, TKinematicJoint *parent, std::list<int> &opened);

};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of namespace kinematics
}  // end of namespace loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // loco_rabbits_kinematics_h
//-------------------------------------------------------------------------------------------
