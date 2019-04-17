//-------------------------------------------------------------------------------------------
/*! \file    kinematics.cpp
    \brief   Kinematics solver
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \date    Nov.20, 2009
*/
//-------------------------------------------------------------------------------------------
#include <lora/octave.h>
//-------------------------------------------------------------------------------------------
#include "kinematics.h"
#include <lora/octave.h>
#include <lora/stl_math.h>
#include <lora/type_gen_oct.h>
#include <fstream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
namespace kinematics
{
// using namespace std;
// using namespace boost;


//-------------------------------------------------------------------------------------------
// [-- matrix and vector calculations from ODE
//-------------------------------------------------------------------------------------------

// from "odemath.h"
#define DOP(a,op,b,c) \
    (a)[0] = ((b)[0]) op ((c)[0]); \
    (a)[1] = ((b)[1]) op ((c)[1]); \
    (a)[2] = ((b)[2]) op ((c)[2]);
#define DOPC(a,op,b,c) \
    (a)[0] = ((b)[0]) op (c); \
    (a)[1] = ((b)[1]) op (c); \
    (a)[2] = ((b)[2]) op (c);
#define DOPE(a,op,b) \
    (a)[0] op ((b)[0]); \
    (a)[1] op ((b)[1]); \
    (a)[2] op ((b)[2]);

#define DDOTpq(a,b,p,q) ((a)[0]*(b)[0] + (a)[p]*(b)[q] + (a)[2*(p)]*(b)[2*(q)])

inline DReal DDOT   (const DReal *a, const DReal *b) { return DDOTpq(a,b,1,1); }
inline DReal DDOT14 (const DReal *a, const DReal *b) { return DDOTpq(a,b,1,4); }

#define DCROSS(a,op,b,c) \
do { \
  (a)[0] op ((b)[1]*(c)[2] - (b)[2]*(c)[1]); \
  (a)[1] op ((b)[2]*(c)[0] - (b)[0]*(c)[2]); \
  (a)[2] op ((b)[0]*(c)[1] - (b)[1]*(c)[0]); \
} while(0)

#define DMULTIPLYOP0_331(A,op,B,C) \
do { \
  (A)[0] op DDOT((B),(C)); \
  (A)[1] op DDOT((B+4),(C)); \
  (A)[2] op DDOT((B+8),(C)); \
} while(0)
#define DMULTIPLYOP0_333(A,op,B,C) \
do { \
  (A)[0] op DDOT14((B),(C)); \
  (A)[1] op DDOT14((B),(C+1)); \
  (A)[2] op DDOT14((B),(C+2)); \
  (A)[4] op DDOT14((B+4),(C)); \
  (A)[5] op DDOT14((B+4),(C+1)); \
  (A)[6] op DDOT14((B+4),(C+2)); \
  (A)[8] op DDOT14((B+8),(C)); \
  (A)[9] op DDOT14((B+8),(C+1)); \
  (A)[10] op DDOT14((B+8),(C+2)); \
} while(0)

#define DECL template <class TA, class TB, class TC> inline void
DECL DMULTIPLY0_331(TA *A, const TB *B, const TC *C) { DMULTIPLYOP0_331(A,=,B,C); }
DECL DMULTIPLY0_333(TA *A, const TB *B, const TC *C) { DMULTIPLYOP0_333(A,=,B,C); }


// from "rotation.cpp"
typedef DReal DQuaternion[4];
#define _R(i,j) R[(i)*4+(j)]

void DQFromAxisAndAngle (DQuaternion q, DReal ax, DReal ay, DReal az,
                         DReal angle)
{
  DReal l = ax*ax + ay*ay + az*az;
  if (l > (0.0)) {
    angle *= (0.5);
    q[0] = std::cos (angle);
    l = std::sin(angle) * (1.0/std::sqrt(l));
    q[1] = ax*l;
    q[2] = ay*l;
    q[3] = az*l;
  }
  else {
    q[0] = 1;
    q[1] = 0;
    q[2] = 0;
    q[3] = 0;
  }
}

void DRfromQ (DMatrix3 R, const DQuaternion q)
{
  // q = (s,vx,vy,vz)
  DReal qq1 = 2*q[1]*q[1];
  DReal qq2 = 2*q[2]*q[2];
  DReal qq3 = 2*q[3]*q[3];
  _R(0,0) = 1 - qq2 - qq3;
  _R(0,1) = 2*(q[1]*q[2] - q[0]*q[3]);
  _R(0,2) = 2*(q[1]*q[3] + q[0]*q[2]);
  _R(0,3) = (0.0);
  _R(1,0) = 2*(q[1]*q[2] + q[0]*q[3]);
  _R(1,1) = 1 - qq1 - qq3;
  _R(1,2) = 2*(q[2]*q[3] - q[0]*q[1]);
  _R(1,3) = (0.0);
  _R(2,0) = 2*(q[1]*q[3] - q[0]*q[2]);
  _R(2,1) = 2*(q[2]*q[3] + q[0]*q[1]);
  _R(2,2) = 1 - qq1 - qq2;
  _R(2,3) = (0.0);
}

inline void DRFromAxisAndAngle (DMatrix3 R, DReal ax, DReal ay, DReal az,
                         DReal angle)
{
  DQuaternion q;
  DQFromAxisAndAngle (q,ax,ay,az,angle);
  DRfromQ(R,q);
}


//-------------------------------------------------------------------------------------------
// end of matrix and vector calculations from ODE --]
//-------------------------------------------------------------------------------------------


//! angle[rad] --> angle[rad] in [-M_PI,M_PI]
template< class T >
inline void ReviseAngle (T& ang)
{
  T tmp = std::fmod(ang,static_cast<T>(2.0l*M_PIl));
  ang= (tmp >= 0.0l) ? ((tmp>M_PIl)?(tmp-2.0l*M_PIl):tmp) : ((tmp>-M_PIl)?tmp:(tmp+2.0l*M_PIl));
}

inline void CopyVec (DVector3 dest, const DVector3 src)
{
  memcpy (dest, src, sizeof(DVector3));
}
inline void CopyMat (DMatrix3 dest, const DMatrix3 src)
{
  memcpy (dest, src, sizeof(DMatrix3));
}
//-------------------------------------------------------------------------------------------

static void Rot2Omega (const TRealMatrix &R, TRealVector &w, const double &eps_=1.0e-6)
{
  double alpha= (R(0,0)+R(1,1)+R(2,2) - 1.0) / 2.0;;

  if((alpha-1.0 < eps_) && (alpha-1.0 > -eps_))  w.fill(0.0);
  else
  {
    double th = std::acos(alpha);
    double tmp= 0.5 * th / std::sin(th);
    w(0) = tmp * (R(2,1) - R(1,2));
    w(1) = tmp * (R(0,2) - R(2,0));
    w(2) = tmp * (R(1,0) - R(0,1));
  }
}
//-------------------------------------------------------------------------------------------

//!\todo FIXME: this routine requires HIGH COMPUTATIONAL COST!!
static TRealMatrix DMat2Oct (const DMatrix3 R)
{
  // copied from line 34 of file rotation.cpp
  #define R_(i,j) R[(i)*4+(j)]
  TRealMatrix res(3,3);
  res(0,0)=R_(0,0); res(0,1)=R_(0,1); res(0,2)=R_(0,2);
  res(1,0)=R_(1,0); res(1,1)=R_(1,1); res(1,2)=R_(1,2);
  res(2,0)=R_(2,0); res(2,1)=R_(2,1); res(2,2)=R_(2,2);
  return res;
  #undef R_
}
//-------------------------------------------------------------------------------------------

//!\brief detect singularity and try to revise it
static void ReviseJacobian (TRealMatrix &J, const double &th=1.0e-3)
{
  const int rows(J.rows()), cols(J.cols());
  double sqsum;
  for(int r(0); r<rows; ++r)
  {
    sqsum= 0.0;
    for(int c(0); c<cols; ++c)
      sqsum+= Square(J(r,c));
    if (sqsum<Square(th))  // true : row r is singular
    {
      J.fill (0.0, r,0, r,cols-1);
      std::cerr<<"Jacobian is singular at row "<<r<<", sqsum= "<<sqsum<<std::endl;
    }
  }
}
//-------------------------------------------------------------------------------------------



//===========================================================================================
// class TKinematicRobot
//===========================================================================================

//!\brief clear the model
void TKinematicRobot::Clear (void)
{
  links_.clear();
  joints_.clear();
  linkid_to_idx_.clear();
  jointid_to_idx_.clear();
  anglevec_idx_to_q_ptr_.clear();
  movable_jointptr_to_anglevec_idx_.clear();
  end_effector_pos_list_.clear();
  end_effector_rot_list_.clear();
  end_effector_list_is_changed_= true;
}
//-------------------------------------------------------------------------------------------

//!\brief add movable joint that is used to solve inverse kinematics
void TKinematicRobot::AddMovableJoint (const TJointID &jointid)
{
  if (jointid_to_idx_.find(jointid)==jointid_to_idx_.end())
  {
    LERROR("joint "<<jointid<<" does not exist.");
    return;
  }
  TKinematicJoint  &joint (joints_[jointid_to_idx_[jointid]]);
  movable_jointptr_to_anglevec_idx_[&joint]=  anglevec_idx_to_q_ptr_.size();

  anglevec_idx_to_q_ptr_.push_back (&joint.q1);
  if (joint.Type==jtUniversal)  anglevec_idx_to_q_ptr_.push_back (&joint.q2);
}
//-------------------------------------------------------------------------------------------

//!\brief clear movable joints
void TKinematicRobot::ClearMovableJoints ()
{
  anglevec_idx_to_q_ptr_.clear();
  movable_jointptr_to_anglevec_idx_.clear();
}
//-------------------------------------------------------------------------------------------

//!\brief add an end effector (position)
void TKinematicRobot::AddEndEffectorPos (const TLinkID &linkid)
{
  if (linkid_to_idx_.find(linkid)==linkid_to_idx_.end())
  {
    LERROR("link "<<linkid<<" cannot be an end-effector.");
    return;
  }
  // TODEIKEndEffectorPos(&links_[linkid_to_idx_[link]])
  end_effector_pos_list_.push_back(linkid_to_idx_[linkid]);
  end_effector_list_is_changed_= true;
}
//-------------------------------------------------------------------------------------------

//!\brief add an end effector (rotation)
void TKinematicRobot::AddEndEffectorRot (const TLinkID &linkid)
{
  if (linkid_to_idx_.find(linkid)==linkid_to_idx_.end())
  {
    LERROR("link "<<linkid<<" cannot be an end-effector.");
    return;
  }
  end_effector_rot_list_.push_back(linkid_to_idx_[linkid]);
  end_effector_list_is_changed_= true;
}
//-------------------------------------------------------------------------------------------

//!\brief clear every end effectors
void TKinematicRobot::ClearEndEffectors ()
{
  end_effector_pos_list_.clear();
  end_effector_rot_list_.clear();
  end_effector_list_is_changed_= true;
}
//-------------------------------------------------------------------------------------------

//!\brief set the target position of the link
void TKinematicRobot::SetEndEffectorTargetPos (const TLinkID &linkid, DVector3 target_pos)
{
  if (linkid_to_idx_.find(linkid)==linkid_to_idx_.end())
  {
    LERROR("link "<<linkid<<" does not exist.");
    return;
  }
  CopyVec (links_[linkid_to_idx_[linkid]].TargetPos, target_pos);
}
//-------------------------------------------------------------------------------------------

//!\brief set the target rotation of the link
void TKinematicRobot::SetEndEffectorTargetRot (const TLinkID &linkid, DMatrix3 target_rot)
{
  if (linkid_to_idx_.find(linkid)==linkid_to_idx_.end())
  {
    LERROR("link "<<linkid<<" does not exist.");
    return;
  }
  CopyMat (links_[linkid_to_idx_[linkid]].TargetRot, target_rot);
}
//-------------------------------------------------------------------------------------------


//!\brief execute inverse kinematics (from current pose)
DReal TKinematicRobot::ExecInverseKinematics (const DReal &tol, int MaxItr, const DReal &step_size, TInverseType inv_type)
{
std::ofstream ofs("debug.dat");
  // if (end_effector_list_is_changed_)

  // Apply the current joint angles and compute the forward kinematics
  reset_to_base_();
  set_joint_angles_(&(links_[base_link_idx_]), NULL);

  int count(0);
  err_vector_.resize (end_effector_pos_list_.size()*3+end_effector_rot_list_.size()*3);
  get_error_vector_ (OctBegin(err_vector_));
  DReal err(0.0);
  do
  {
    get_current_jacobian_ (J_);
    // LDEBUG("ReviseJacobian...");
    ReviseJacobian(J_);
    // for (int r(0),rmax(std::min(J_.rows(),J_.cols())); r<rmax; ++r)
      // J_(r,r)+=0.001;
    // LDEBUG("J_="<<endl<<J_);
    // LDEBUG("J_.pseudo_inverse()="<<endl<<J_.pseudo_inverse());
    // LDBGVAR(err_vector_.transpose());
    switch(inv_type)
    {
      case itPseudoInverse :  dq_= J_.pseudo_inverse() * err_vector_; break;
      case itTranspose     :  dq_= J_.transpose() * err_vector_; break;
      default :  LERROR("in ExecInverseKinematics, invalid inv_type: "<<inv_type); lexit(df);
    }
    loco_rabbits::operator*=(dq_, step_size);
    add_to_joint_angles_ (&(links_[base_link_idx_]), NULL, OctBegin(dq_));
for (int j(0), jmax(anglevec_idx_to_q_ptr_.size()); j<jmax; ++j)
  ofs<<" "<<(*anglevec_idx_to_q_ptr_[j]);
ofs<< " # "<<err_vector_.transpose()<<std::endl;
    get_error_vector_ (OctBegin(err_vector_));
    err= GetNorm(err_vector_);
    ++count;
  } while (err>tol && count<MaxItr);
  // LDEBUG(count<<"-err:  "<<err);
ofs<< " #@ "<<count<<" "<<err<<std::endl;
  return err;
}
//-------------------------------------------------------------------------------------------


int TKinematicRobot::AddLink (const TKinematicLink &link)
{
  links_.push_back(link);
  int idx= links_.size()-1;
  linkid_to_idx_[link.LinkID]= idx;
  return idx;
}
//-------------------------------------------------------------------------------------------
int TKinematicRobot::AddJoint (const TKinematicJoint &joint)
{
  joints_.push_back(joint);
  int idx= joints_.size()-1;
  jointid_to_idx_[joint.JointID]= idx;
  return idx;
}
//-------------------------------------------------------------------------------------------

void TKinematicRobot::AddConnectedJointToLink (int link_idx, int joint_idx)
{
  links_[link_idx].Connected.push_back (joint_idx);
}
//-------------------------------------------------------------------------------------------

//!\brief set ChangeableLinks
void TKinematicRobot::SetChangeableLinks (void)
{
  for (TKinematicJointSet::iterator jitr(joints_.begin()); jitr!=joints_.end(); ++jitr)
  {
    jitr->ChangeableLinks.clear();
    jitr->ChangeableLinks.resize(links_.size(),false);
  }
  std::list<int> opened;
  set_changeable_links_ (&links_[base_link_idx_],NULL,opened);
}
//-------------------------------------------------------------------------------------------


/*! \brief rotate all succeeding links connected to ib in rotation rot around pos */
void TKinematicRobot::rot_succeeding_links_ (TKinematicLink *ib, TKinematicJoint *parent, const DVector3 pos, const DMatrix3 rot)
{
  DVector3 c, d;
  CopyVec (d, ib->p);
  DOPE(d,-=,pos);
  DMULTIPLY0_331 (c,rot,d);  // c = rot * d
  DOPE(c,+=,pos);
  DMatrix3 R;
  DMULTIPLY0_333 (R,rot, ib->R);  // R = rot * ib->R
  CopyVec (ib->p, c);
  CopyMat (ib->R, R);

  for (TJointList::iterator jitr(ib->Connected.begin()); jitr!=ib->Connected.end(); ++jitr)
  {
    if(&joints_[*jitr]==parent) continue;

    CopyVec (d, joints_[*jitr].Anchor);
    DOPE(d,-=,pos);
    DMULTIPLY0_331 (c,rot,d);  // c = rot * d
    DOPE(c,+=,pos);
    CopyVec (joints_[*jitr].Anchor, c);
    DMULTIPLY0_331 (c,rot,joints_[*jitr].Axis1);  // c = rot * d
    CopyVec (joints_[*jitr].Axis1, c);
    DMULTIPLY0_331 (c,rot,joints_[*jitr].Axis2);  // c = rot * d
    CopyVec (joints_[*jitr].Axis2, c);

    TKinematicLink *ib2= &links_[joints_[*jitr].B1];
    if(ib2==ib) ib2= &links_[joints_[*jitr].B2];
    if(ib2==ib) {LERROR("fatal!"); lexit(df);}
    rot_succeeding_links_(ib2, &joints_[*jitr], pos, rot);
  }
}
//-------------------------------------------------------------------------------------------

/*! \brief move all succeeding links connected to ib in distance d */
void TKinematicRobot::move_succeeding_links_ (TKinematicLink *ib, TKinematicJoint *parent, const DVector3 d)
{
  DOPE(ib->p,+=,d);
  // for each link connected to ib
  for (TJointList::iterator jitr(ib->Connected.begin()); jitr!=ib->Connected.end(); ++jitr)
  {
    if(&joints_[*jitr]==parent) continue;

    DOPE(joints_[*jitr].Anchor,+=,d);

    TKinematicLink *ib2= &links_[joints_[*jitr].B1];
    if(ib2==ib) ib2= &links_[joints_[*jitr].B2];
    if(ib2==ib) {LERROR("fatal!"); lexit(df);}
    move_succeeding_links_(ib2, &joints_[*jitr], d);
  }
}
//-------------------------------------------------------------------------------------------

//!\brief reset current p,R,Anchor,Axis1,Axis2 to base line.
void TKinematicRobot::reset_to_base_ ()
{
  for (TKinematicLinkSet::iterator itr(links_.begin()); itr!=links_.end(); ++itr)
  {
    CopyMat (itr->R, itr->BaseRot);
    CopyVec (itr->p, itr->BasePos);
  }
  for (TKinematicJointSet::iterator itr(joints_.begin()); itr!=joints_.end(); ++itr)
  {
    if (itr->Type==jtHinge)
    {
      CopyVec (itr->Anchor, itr->BaseAnchor);
      CopyVec (itr->Axis1,  itr->BaseAxis1);
    }
    else if (itr->Type==jtSlider)
    {
      CopyVec (itr->Axis1,  itr->BaseAxis1);
    }
    else if (itr->Type==jtUniversal)
    {
      CopyVec (itr->Anchor, itr->BaseAnchor);
      CopyVec (itr->Axis1,  itr->BaseAxis1);
      CopyVec (itr->Axis2,  itr->BaseAxis2);
    }
    // else if (itr->Type==jtHinge2)
    // {
    // }
    else
    {
      LWARNING("in reset_to_base_, not implemented joint type ("<<itr->Type<<")");
    }
  }
}
//-------------------------------------------------------------------------------------------

//!\brief set joint angles (or positions) which are set to q1/q2 (robot is rotated; forward kinematics)
void TKinematicRobot::set_joint_angles_ (TKinematicLink *ib, TKinematicJoint *parent)
{
  if (ib==NULL)  return;
  for (TJointList::iterator jitr(ib->Connected.begin()); jitr!=ib->Connected.end(); ++jitr)
  {
    if(&joints_[*jitr]==parent) continue;

    DReal  sign (1.0);  //! \TODO FIXME check sign and make it correct!!!!
    TKinematicLink *ib2= &links_[joints_[*jitr].B1];
    if(ib2==ib) {ib2= &links_[joints_[*jitr].B2]; sign=-1.0;}
    if(ib2==ib) {LERROR("fatal!"); lexit(df);}

    TMovableJointMap::const_iterator  itr_anglevec_idx= movable_jointptr_to_anglevec_idx_.find(&joints_[*jitr]);
    if (itr_anglevec_idx != movable_jointptr_to_anglevec_idx_.end())
    {
      // const int i1= itr_anglevec_idx->second;
      // const int i2= i1+1;
      if (joints_[*jitr].Type==jtHinge)
      {
        if (joints_[*jitr].q1 > joints_[*jitr].Hi1)  joints_[*jitr].q1 = joints_[*jitr].Hi1;
        if (joints_[*jitr].q1 < joints_[*jitr].Lo1)  joints_[*jitr].q1 = joints_[*jitr].Lo1;
        ReviseAngle (joints_[*jitr].q1);
        DReal *a(joints_[*jitr].Axis1), *p(joints_[*jitr].Anchor);
        DMatrix3 rotR;
        DRFromAxisAndAngle (rotR,a[0],a[1],a[2], sign*(joints_[*jitr].q1-joints_[*jitr].Offset1));
        rot_succeeding_links_(ib2, &joints_[*jitr], p, rotR);
      }
      else if (joints_[*jitr].Type==jtSlider)
      {
        if (joints_[*jitr].q1 > joints_[*jitr].Hi1)  joints_[*jitr].q1 = joints_[*jitr].Hi1;
        if (joints_[*jitr].q1 < joints_[*jitr].Lo1)  joints_[*jitr].q1 = joints_[*jitr].Lo1;
        DVector3 d;
        DOPC(d,*,joints_[*jitr].Axis1, sign*(joints_[*jitr].q1-joints_[*jitr].Offset1));
        move_succeeding_links_(ib2, &joints_[*jitr], d);
      }
      else if (joints_[*jitr].Type==jtUniversal)
      {
        if (joints_[*jitr].q1 > joints_[*jitr].Hi1)  joints_[*jitr].q1 = joints_[*jitr].Hi1;
        if (joints_[*jitr].q1 < joints_[*jitr].Lo1)  joints_[*jitr].q1 = joints_[*jitr].Lo1;
        if (joints_[*jitr].q2 > joints_[*jitr].Hi2)  joints_[*jitr].q2 = joints_[*jitr].Hi2;
        if (joints_[*jitr].q2 < joints_[*jitr].Lo2)  joints_[*jitr].q2 = joints_[*jitr].Lo2;
        ReviseAngle (joints_[*jitr].q1);
        ReviseAngle (joints_[*jitr].q2);
        DReal *a, *p;
        DMatrix3 rotR;
        p= joints_[*jitr].Anchor;
        a= joints_[*jitr].Axis1;
        DRFromAxisAndAngle (rotR,a[0],a[1],a[2], sign*(joints_[*jitr].q1-joints_[*jitr].Offset1));
        rot_succeeding_links_(ib2, &joints_[*jitr], p, rotR);
        DVector3 ax2;
        DMULTIPLY0_331 (ax2,rotR,joints_[*jitr].Axis2);  // c = rot * d
        CopyVec (joints_[*jitr].Axis2, ax2);
        a= joints_[*jitr].Axis2;
        DRFromAxisAndAngle (rotR,a[0],a[1],a[2], sign*(joints_[*jitr].q2-joints_[*jitr].Offset2));
        rot_succeeding_links_(ib2, &joints_[*jitr], p, rotR);
      }
      // else if (dJointGetType(ij)==dJointTypeHinge2)
      // {
      // }
      else
      {
        LWARNING("in set_joint_angles_, not implemented joint type ("<<joints_[*jitr].Type<<")");
      }
    }
    set_joint_angles_(ib2, &joints_[*jitr]);
  }
}
//-------------------------------------------------------------------------------------------

//!\brief add dq to joint angles (robot is rotated; forward kinematics)
void TKinematicRobot::add_to_joint_angles_ (TKinematicLink *ib, TKinematicJoint *parent, const double *dq)
{
  for (int j(0), jmax(anglevec_idx_to_q_ptr_.size()); j<jmax; ++j)
    (*anglevec_idx_to_q_ptr_[j])+= -dq[j];
  reset_to_base_();
  set_joint_angles_(ib,parent);
}
//-------------------------------------------------------------------------------------------

//!\brief calculate Jacobian at the current pose and asign it into J
void TKinematicRobot::get_current_jacobian_ (TRealMatrix &J)
{
  J.resize (end_effector_pos_list_.size()*3+end_effector_rot_list_.size()*3, anglevec_idx_to_q_ptr_.size());
  DVector3  dp, pj;
  int r(0),c(0);
  // for (TKinematicJointSet::const_iterator jitr(joints_.begin()); jitr!=joints_.end(); ++jitr)
  for (TMovableJointMap::const_iterator jmapitr(movable_jointptr_to_anglevec_idx_.begin());
          jmapitr!=movable_jointptr_to_anglevec_idx_.end(); ++jmapitr)
  {
    const TKinematicJoint  &joint (*jmapitr->first);
    r= 0;
    #define SET_J_ZERO  {J(r++,c)=0.0; J(r++,c)=0.0; J(r++,c)=0.0;}
    if (joint.Type==jtHinge)
    {
      for (std::vector<int>::const_iterator eitr(end_effector_pos_list_.begin()); eitr!=end_effector_pos_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          const TKinematicLink &target_link (links_[*eitr]);
          // J= joint.Axis1 x (target_link.p - joint.Anchor)
          DOP(dp,-,target_link.p,joint.Anchor);
          DCROSS (pj,=,joint.Axis1,dp);
          J(r++,c)= pj[0];
          J(r++,c)= pj[1];
          J(r++,c)= pj[2];
        }
        else SET_J_ZERO
      }
      for (std::vector<int>::const_iterator eitr(end_effector_rot_list_.begin()); eitr!=end_effector_rot_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          // J= joint.Axis1
          J(r++,c)= joint.Axis1[0];
          J(r++,c)= joint.Axis1[1];
          J(r++,c)= joint.Axis1[2];
        }
        else SET_J_ZERO
      }
    }
    else if (joint.Type==jtSlider)
    {
      for (std::vector<int>::const_iterator eitr(end_effector_pos_list_.begin()); eitr!=end_effector_pos_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          // J= joint.Axis1
          J(r++,c)= joint.Axis1[0];
          J(r++,c)= joint.Axis1[1];
          J(r++,c)= joint.Axis1[2];
        }
        else SET_J_ZERO
      }
      for (std::vector<int>::const_iterator eitr(end_effector_rot_list_.begin()); eitr!=end_effector_rot_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          // J= 0
          J(r++,c)= 0.0;
          J(r++,c)= 0.0;
          J(r++,c)= 0.0;
        }
        else SET_J_ZERO
      }
    }
    else if (joint.Type==jtUniversal)
    {
      for (std::vector<int>::const_iterator eitr(end_effector_pos_list_.begin()); eitr!=end_effector_pos_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          const TKinematicLink &target_link (links_[*eitr]);
          // J= joint.Axis1 x (target_link.p - joint.Anchor)
          DOP(dp,-,target_link.p,joint.Anchor);
          DCROSS (pj,=,joint.Axis1,dp);
          J(r++,c)= pj[0];
          J(r++,c)= pj[1];
          J(r++,c)= pj[2];
        }
        else SET_J_ZERO
      }
      for (std::vector<int>::const_iterator eitr(end_effector_rot_list_.begin()); eitr!=end_effector_rot_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          // J= joint.Axis1
          J(r++,c)= joint.Axis1[0];
          J(r++,c)= joint.Axis1[1];
          J(r++,c)= joint.Axis1[2];
        }
        else SET_J_ZERO
      }
      ++c;
      r=0;
      for (std::vector<int>::const_iterator eitr(end_effector_pos_list_.begin()); eitr!=end_effector_pos_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          const TKinematicLink &target_link (links_[*eitr]);
          // J= joint.Axis2 x (target_link.p - joint.Anchor)
          DOP(dp,-,target_link.p,joint.Anchor);
          DCROSS (pj,=,joint.Axis2,dp);
          J(r++,c)= pj[0];
          J(r++,c)= pj[1];
          J(r++,c)= pj[2];
        }
        else SET_J_ZERO
      }
      for (std::vector<int>::const_iterator eitr(end_effector_rot_list_.begin()); eitr!=end_effector_rot_list_.end(); ++eitr)
      {
        if (joint.ChangeableLinks[*eitr])
        {
          // J= joint.Axis2
          J(r++,c)= joint.Axis2[0];
          J(r++,c)= joint.Axis2[1];
          J(r++,c)= joint.Axis2[2];
        }
        else SET_J_ZERO
      }
    }
    // else if (dJointGetType(ij)==dJointTypeHinge2)
    // {
    // }
    else
    {
      LWARNING("in get_current_jacobian_, not implemented joint type ("<<joint.Type<<")");
    }
    ++c;
  }
}
//-------------------------------------------------------------------------------------------

//!\brief calculate error vector
void TKinematicRobot::get_error_vector_ (double *err_vector)
{
  Rerr_.resize(3,3,0.0);
  werr_.resize(3,0.0);
  for (std::vector<int>::const_iterator eitr(end_effector_pos_list_.begin()); eitr!=end_effector_pos_list_.end(); ++eitr)
  {
    const TKinematicLink &target_link (links_[*eitr]);
    *(err_vector++)= target_link.TargetPos[0] - target_link.p[0];
    *(err_vector++)= target_link.TargetPos[1] - target_link.p[1];
    *(err_vector++)= target_link.TargetPos[2] - target_link.p[2];
  }
  for (std::vector<int>::const_iterator eitr(end_effector_rot_list_.begin()); eitr!=end_effector_rot_list_.end(); ++eitr)
  {
    const TKinematicLink &target_link (links_[*eitr]);
    target_link_R_= DMat2Oct(target_link.R);
    Rerr_= target_link_R_.inverse() * DMat2Oct(target_link.TargetRot);
    Rot2Omega(Rerr_,werr_);
    werr_= target_link_R_ * werr_;
    *(err_vector++)= werr_(0);
    *(err_vector++)= werr_(1);
    *(err_vector++)= werr_(2);
  }
}
//-------------------------------------------------------------------------------------------

//!\brief set ChangeableLinks
void TKinematicRobot::set_changeable_links_ (TKinematicLink *ib, TKinematicJoint *parent, std::list<int> &opened)
{
  // for each link connected to ib
  for (TJointList::iterator jitr(ib->Connected.begin()); jitr!=ib->Connected.end(); ++jitr)
  {
    if(&joints_[*jitr]==parent) continue;
    int ib2_idx= joints_[*jitr].B1;
    TKinematicLink *ib2= &links_[ib2_idx];
    if(ib2==ib) {ib2_idx=joints_[*jitr].B2;  ib2= &links_[ib2_idx];}
    if(ib2==ib) {LERROR("fatal!"); lexit(df);}

    opened.push_back (*jitr);
    for (std::list<int>::iterator oitr(opened.begin()); oitr!=opened.end(); ++oitr)
      joints_[*oitr].ChangeableLinks[ib2_idx]= true;

    set_changeable_links_(ib2, &joints_[*jitr], opened);
  }
}
//-------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------
}  // end of namespace kinematics
}  // end of namespace loco_rabbits
//-------------------------------------------------------------------------------------------

