#!/usr/bin/python
#\file    kdl_kin2.py
#\brief   Robot kinematics solver using KDL.
#         IK is updated to use weighted pseudo inverse of Jacobian.
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Oct.23, 2017
#\date    Feb.5, 2018
#\date    Jun.12, 2018
import roslib; roslib.load_manifest('urdfdom_py')
import rospy
import kdl_parser_py.urdf
import PyKDL
import numpy as np

'''
PyKDL wrapper class.
Load URDF from robot_description.
based on:
  https://github.com/RethinkRobotics/baxter_pykdl
  https://github.com/akihikoy/baxter_pykdl
  baxter_pykdl/src/baxter_pykdl/baxter_pykdl.py
PyKDL:
  https://github.com/orocos/orocos_kinematics_dynamics
'''
class TKinematics(object):
  '''Create the class.
    base_link Base link of kinematic chain to be considered.
      Can be None; in this case, a root link obtained from URDF is used.
    end_link End link of kinematic chain to be considered. '''
  def __init__(self, base_link=None, end_link=None, description='robot_description'):
    self._robot = kdl_parser_py.urdf.urdf.URDF.from_parameter_server(description)
    (ok, self._kdl_tree)= kdl_parser_py.urdf.treeFromUrdfModel(self._robot)
    self._base_link = self._robot.get_root() if base_link is None else base_link
    self._tip_link = end_link
    self._tip_frame = PyKDL.Frame()
    self._arm_chain = self._kdl_tree.getChain(self._base_link, self._tip_link)

    #self.joint_names = [joint.name for joint in self._robot.joints if joint.type!='fixed']
    self.joint_names = [self._arm_chain.getSegment(i).getJoint().getName() for i in range(self._arm_chain.getNrOfSegments()) if self._arm_chain.getSegment(i).getJoint().getType()!=PyKDL.Joint.None]
    self._num_jnts = len(self.joint_names)

    # Store joint information for future use
    self.get_joint_information()

    # KDL Solvers
    self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
    self._fk_v_kdl = PyKDL.ChainFkSolverVel_recursive(self._arm_chain)
    #self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
    #self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR(self._arm_chain, self._fk_p_kdl, self._ik_v_kdl)
    self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
    self._dyn_kdl = PyKDL.ChainDynParam(self._arm_chain, PyKDL.Vector.Zero())

  def print_robot_description(self):
    print "URDF non-fixed joints: %d;" % len([joint.type for joint in self._robot.joints if joint.type!='fixed'])
    print "URDF total joints: %d" % len(self._robot.joints)
    print "URDF links: %d" % len(self._robot.links)
    print "URDF link names: %s" % [link.name for link in self._robot.links]
    print "URDF joint names: %s" % [joint.name for joint in self._robot.joints]
    print "URDF Root: %s" % self._robot.get_root()

    print "KDL segments: %d" % self._kdl_tree.getNrOfSegments()
    print "KDL joints: %d" % self._kdl_tree.getNrOfJoints()

    print "KDL-chain segments: %d" % self._arm_chain.getNrOfSegments()
    print "KDL-chain joints: %d" % self._arm_chain.getNrOfJoints()
    print "KDL-chain segment names: %s" % [self._arm_chain.getSegment(i).getName() for i in range(self._arm_chain.getNrOfSegments())]
    print "KDL-chain joint names: %s" % [self._arm_chain.getSegment(i).getJoint().getName() for i in range(self._arm_chain.getNrOfSegments())]
    print "KDL-chain joint types: %s" % [self._arm_chain.getSegment(i).getJoint().getType() for i in range(self._arm_chain.getNrOfSegments())]
    #print [self._arm_chain.getSegment(i) for i in range(self._arm_chain.getNrOfSegments())]

    print "Effective joint names: %s" % self.joint_names

  def get_joint_information(self):
    self._urdf_joints = {joint.name:joint for joint in self._robot.joints if joint.type!='fixed'}
    limits= [self._urdf_joints[jnt_name].limit for jnt_name in self.joint_names]
    self.joint_limits_lower = [-np.inf if (limit is None or limit.lower is None) else limit.lower for limit in limits]
    self.joint_limits_upper = [+np.inf if (limit is None or limit.lower is None) else limit.upper for limit in limits]
    self.joint_types = [self._urdf_joints[jnt_name].type for jnt_name in self.joint_names]

  def joints_to_kdl(self, type, values=None):
    kdl_array = PyKDL.JntArray(self._num_jnts)

    if values is None:
        raise Exception('Error in TKinematics.joints_to_kdl')
        #if type == 'positions':
            #cur_type_values = self._limb_interface.joint_angles()
        #elif type == 'velocities':
            #cur_type_values = self._limb_interface.joint_velocities()
        #elif type == 'torques':
            #cur_type_values = self._limb_interface.joint_efforts()
    else:
        cur_type_values = values

    for idx, name in enumerate(self.joint_names):
        kdl_array[idx] = cur_type_values[name]
    if type == 'velocities':
        kdl_array = PyKDL.JntArrayVel(kdl_array)
    return kdl_array

  def kdl_to_mat(self, data):
    mat =  np.mat(np.zeros((data.rows(), data.columns())))
    for i in range(data.rows()):
        for j in range(data.columns()):
            mat[i,j] = data[i,j]
    return mat

  def forward_position_kinematics(self, joint_values=None, segment=-1):
    end_frame = PyKDL.Frame()
    self._fk_p_kdl.JntToCart(self.joints_to_kdl('positions',joint_values), end_frame, segment)
    pos = end_frame.p
    rot = PyKDL.Rotation(end_frame.M)
    rot = rot.GetQuaternion()
    return np.array([pos[0], pos[1], pos[2],  rot[0], rot[1], rot[2], rot[3]])

  def forward_velocity_kinematics(self,joint_velocities=None):
    end_frame = PyKDL.FrameVel()
    self._fk_v_kdl.JntToCart(self.joints_to_kdl('velocities',joint_velocities), end_frame)
    return end_frame.GetTwist()

  '''
  IK interface.
    position, orientation: Target pose.
    seed: Initial joint positions.
    min_joints,max_joints: Joint limits.  If None, existing values are used.
    w_x, w_q: ChainIkSolverVel_wdls (weighted pinv) is used if one of them is not None.
      cf. https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/chainiksolvervel_wdls.hpp
      w_x: Weights on task space (position and orientation).
      w_q: Weights on joint space (joint positions).
      w_x and w_q should be a list of matrix, e.g. np.diag([1.0,1.0,1.0, 0.1,0.1,0.1]).tolist()
    maxiter: Number of max iterations.
    eps: Tolerance.
    with_st: Return with state (state = If-solved, last-result).
      If False, this returns: last-result (solved), or, None (not solved).
  TODO:
    We want to use ChainIkSolverPos_LMA (IK with Levenberg-Marquardt) which is more robust.
    However the current implementation does not take into account the joint limits.
  '''
  def inverse_kinematics(self, position, orientation=None, seed=None, min_joints=None, max_joints=None, w_x=None, w_q=None, maxiter=500, eps=1.0e-6, with_st=False):
    if w_x is None and w_q is None:
      ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
    else:
      ik_v_kdl = PyKDL.ChainIkSolverVel_wdls(self._arm_chain)
      if w_x is not None:  ik_v_kdl.setWeightTS(w_x)  #TS = Task Space
      if w_q is not None:  ik_v_kdl.setWeightJS(w_q)  #JS = Joint Space
    pos = PyKDL.Vector(position[0], position[1], position[2])
    if orientation is not None:
        rot = PyKDL.Rotation()
        rot = rot.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
    # Populate seed with current angles if not provided
    seed_array = PyKDL.JntArray(self._num_jnts)
    if seed is not None:
        seed_array.resize(len(seed))
        for idx, jnt in enumerate(seed):
            seed_array[idx] = jnt
    else:
        seed_array = self.joints_to_kdl('positions')

    # Make IK Call
    if orientation is not None:
        goal_pose = PyKDL.Frame(rot, pos)
    else:
        goal_pose = PyKDL.Frame(pos)
    result_angles = PyKDL.JntArray(self._num_jnts)

    # Make IK solver with joint limits
    if min_joints is None: min_joints = self.joint_limits_lower
    if max_joints is None: max_joints = self.joint_limits_upper
    mins_kdl = PyKDL.JntArray(len(min_joints))
    for idx,jnt in enumerate(min_joints):  mins_kdl[idx] = jnt
    maxs_kdl = PyKDL.JntArray(len(max_joints))
    for idx,jnt in enumerate(max_joints):  maxs_kdl[idx] = jnt
    ik_p_kdl = PyKDL.ChainIkSolverPos_NR_JL(self._arm_chain, mins_kdl, maxs_kdl,
                                            self._fk_p_kdl, ik_v_kdl, maxiter, eps)

    if ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
        result = np.array(list(result_angles))
        if with_st: return True,result
        else:  return result
    else:
        if with_st:
          result = np.array(list(result_angles))
          return False,result
        else:  return None

  def jacobian(self,joint_values=None):
    jacobian = PyKDL.Jacobian(self._num_jnts)
    self._jac_kdl.JntToJac(self.joints_to_kdl('positions',joint_values), jacobian)
    return self.kdl_to_mat(jacobian)

  def jacobian_transpose(self,joint_values=None):
    return self.jacobian(joint_values).T

  def jacobian_pseudo_inverse(self,joint_values=None):
    return np.linalg.pinv(self.jacobian(joint_values))


  def inertia(self,joint_values=None):
    inertia = PyKDL.JntSpaceInertiaMatrix(self._num_jnts)
    self._dyn_kdl.JntToMass(self.joints_to_kdl('positions',joint_values), inertia)
    return self.kdl_to_mat(inertia)

  def cart_inertia(self,joint_values=None):
    js_inertia = self.inertia(joint_values)
    jacobian = self.jacobian(joint_values)
    return np.linalg.inv(jacobian * np.linalg.inv(js_inertia) * jacobian.T)


if __name__=='__main__':
  print 'Testing TKinematics (robot_description == Yaskawa Motoman is assumed).'
  print 'Before executing this script, run:'
  print '  rosparam load `rospack find motoman_sia10f_support`/urdf/sia10f.urdf robot_description'
  kin= TKinematics(end_link='link_t')
  kin.print_robot_description()

  q0= [0.0]*7
  angles= {joint:q0[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x0= kin.forward_position_kinematics(angles)
  print 'q0=',q0
  print 'x0= FK(q0)=',x0

  import random
  q1= [3.0*(random.random()-0.5) for j in range(7)]
  angles= {joint:q1[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
  x1= kin.forward_position_kinematics(angles)
  print 'q1=',q1
  print 'x1= FK(q1)=',x1

  seed= [0.0]*7
  #seed= [3.0*(random.random()-0.5) for j in range(7)]
  q2= kin.inverse_kinematics(x1[:3], x1[3:], seed=seed, maxiter=2000, eps=1.0e-4)  #, maxiter=500, eps=1.0e-6
  print 'q2= IK(x1)=',q2
  if q2 is not None:
    angles= {joint:q2[j] for j,joint in enumerate(kin.joint_names)}  #Deserialize
    x2= kin.forward_position_kinematics(angles)
    print 'x2= FK(q2)=',x2
    print 'x2==x1?', np.allclose(x2,x1)
    print '|x2-x1|=',np.linalg.norm(x2-x1)
  else:
    print 'Failed to solve IK.'
