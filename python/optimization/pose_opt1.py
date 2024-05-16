#!/usr/bin/python
#\file    pose_opt1.py
#\brief   Optimization of 6d pose (xyz+quaternion).
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    May.16, 2024

import math
import sys
import numpy
import numpy as np
import numpy.linalg as la


#================Copied from ay_py.core._rostf================

def _rostf_identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)
    >>> numpy.allclose(I, numpy.identity(4, dtype=numpy.float64))
    True

    """
    return numpy.identity(4, dtype=numpy.float64)

def _rostf_quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def _rostf_quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = numpy.empty((4, ), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    t = numpy.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

#================Copied from ay_py.core.geom================

##Quaternion to 3x3 rotation matrix
#def QToRot(q):
  #return _rostf_quaternion_matrix(q)[:3,:3]
#Quaternion to 3x3 rotation matrix
#cf. http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
def QToRot(q):
  R= np.array([[0.0]*3]*3)
  qx= q[0]
  qy= q[1]
  qz= q[2]
  qw= q[3]
  sqw = qw*qw
  sqx = qx*qx
  sqy = qy*qy
  sqz = qz*qz

  #invs (inverse square length) is only required if quaternion is not already normalised
  invs = 1.0 / (sqx + sqy + sqz + sqw)
  R[0,0] = ( sqx - sqy - sqz + sqw)*invs  #since sqw + sqx + sqy + sqz =1/invs*invs
  R[1,1] = (-sqx + sqy - sqz + sqw)*invs
  R[2,2] = (-sqx - sqy + sqz + sqw)*invs

  tmp1 = qx*qy
  tmp2 = qz*qw
  R[1,0] = 2.0 * (tmp1 + tmp2)*invs
  R[0,1] = 2.0 * (tmp1 - tmp2)*invs

  tmp1 = qx*qz
  tmp2 = qy*qw
  R[2,0] = 2.0 * (tmp1 - tmp2)*invs
  R[0,2] = 2.0 * (tmp1 + tmp2)*invs
  tmp1 = qy*qz
  tmp2 = qx*qw
  R[2,1] = 2.0 * (tmp1 + tmp2)*invs
  R[1,2] = 2.0 * (tmp1 - tmp2)*invs
  return R

def GetWedge(w):
  wedge= np.zeros((3,3))
  wedge[0,0]=0.0;    wedge[0,1]=-w[2];  wedge[0,2]=w[1]
  wedge[1,0]=w[2];   wedge[1,1]=0.0;    wedge[1,2]=-w[0]
  wedge[2,0]=-w[1];  wedge[2,1]=w[0];   wedge[2,2]=0.0
  return wedge

#Rodrigues formula to get R from w (=angle*axis) where angle is in radian and axis is 3D unit vector.
#NOTE: This function is equivalent to RFromAxisAngle(axis,angle).
def Rodrigues(w, epsilon=1.0e-6):
  th= la.norm(w)
  if th<epsilon:  return np.identity(3)
  w_wedge= GetWedge(np.array(w) *(1.0/th))
  return np.identity(3) + w_wedge * math.sin(th) + np.dot(w_wedge,w_wedge) * (1.0-math.cos(th))

#Inverse of Rodrigues, i.e. returns w (=angle*axis) from R where angle is in radian and axis is 3D unit vector.
#With singularity detection. Src: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
def InvRodrigues(R, epsilon=1.0e-6):
  epsilon2= epsilon
  if abs(R[0,1]-R[1,0])<epsilon and abs(R[0,2]-R[2,0])<epsilon and abs(R[1,2]-R[2,1])<epsilon:
    #singularity found
    #first check for identity matrix which must have +1 for all terms
    #in leading diagonaland zero in other terms
    if abs(R[0,1]+R[1,0])<epsilon2 and abs(R[0,2]+R[2,0])<epsilon2 and abs(R[1,2]+R[2,1])<epsilon2 and abs(R[0,0]+R[1,1]+R[2,2]-3)<epsilon2:
      #this singularity is identity matrix so angle = 0
      return np.zeros(3)
    #otherwise this singularity is angle = 180
    angle= np.pi
    xx= (R[0,0]+1.)/2.
    yy= (R[1,1]+1.)/2.
    zz= (R[2,2]+1.)/2.
    xy= (R[0,1]+R[1,0])/4.
    xz= (R[0,2]+R[2,0])/4.
    yz= (R[1,2]+R[2,1])/4.
    if xx>yy and xx>zz:
      #R[0,0] is the largest diagonal term
      if xx<epsilon:
        x,y,z= 0, np.cos(np.pi/4.), np.cos(np.pi/4.)
      else:
        x= np.sqrt(xx)
        y,z= xy/x, xz/x
    elif yy > zz:
      #R[1,1] is the largest diagonal term
      if yy<epsilon:
        x,y,z= np.cos(np.pi/4.), 0.0, np.cos(np.pi/4.)
      else:
        y= np.sqrt(yy)
        x,z= xy/y, yz/y
    else:
      #R[2,2] is the largest diagonal term so base result on this
      if zz<epsilon:
        x,y,z= np.cos(np.pi/4.), np.cos(np.pi/4.), 0.0
      else:
        z= np.sqrt(zz)
        x,y= xz/z, yz/z
    return angle*np.array([x,y,z])
  #as we have reached here there are no singularities so we can handle normally
  s= np.sqrt((R[2,1]-R[1,2])*(R[2,1]-R[1,2])+(R[0,2]-R[2,0])*(R[0,2]-R[2,0])+(R[1,0]-R[0,1])*(R[1,0]-R[0,1]))
  if np.abs(s)<epsilon:  s=1.0
  #prevent divide by zero, should not happen if matrix is orthogonal and should be
  #caught by singularity test above, but I've left it in just in case
  angle= np.arccos((R[0,0]+R[1,1]+R[2,2]-1.)/2.)
  tmp= angle/s
  return tmp*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

#Get difference of two poses [dx,dy,dz, dwx,dwy,dwz] (intuitively, x2-x1)
def DiffX(x1, x2):
  w= InvRodrigues(np.dot(QToRot(x2[3:]),QToRot(x1[3:]).T))
  return [x2[0]-x1[0],x2[1]-x1[1],x2[2]-x1[2], w[0],w[1],w[2]]

##Quaternion to 3x3 rotation matrix
#def QToRot(q):
  #return _rostf_quaternion_matrix(q)[:3,:3]
#Quaternion to 3x3 rotation matrix
#cf. http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
def QToRot(q):
  R= np.array([[0.0]*3]*3)
  qx= q[0]
  qy= q[1]
  qz= q[2]
  qw= q[3]
  sqw = qw*qw
  sqx = qx*qx
  sqy = qy*qy
  sqz = qz*qz

  #invs (inverse square length) is only required if quaternion is not already normalised
  invs = 1.0 / (sqx + sqy + sqz + sqw)
  R[0,0] = ( sqx - sqy - sqz + sqw)*invs  #since sqw + sqx + sqy + sqz =1/invs*invs
  R[1,1] = (-sqx + sqy - sqz + sqw)*invs
  R[2,2] = (-sqx - sqy + sqz + sqw)*invs

  tmp1 = qx*qy
  tmp2 = qz*qw
  R[1,0] = 2.0 * (tmp1 + tmp2)*invs
  R[0,1] = 2.0 * (tmp1 - tmp2)*invs

  tmp1 = qx*qz
  tmp2 = qy*qw
  R[2,0] = 2.0 * (tmp1 - tmp2)*invs
  R[0,2] = 2.0 * (tmp1 + tmp2)*invs
  tmp1 = qy*qz
  tmp2 = qx*qw
  R[2,1] = 2.0 * (tmp1 + tmp2)*invs
  R[1,2] = 2.0 * (tmp1 - tmp2)*invs
  return R

#3x3 rotation matrix to quaternion
def RotToQ(R):
  M = _rostf_identity_matrix()
  M[:3,:3] = R
  return _rostf_quaternion_from_matrix(M)

#Convert a pose, x,y,z,quaternion(qx,qy,qz,qw) to pos (x,y,z) and 3x3 rotation matrix
def XToPosRot(x):
  p = np.array(x[0:3])
  #R = _rostf_quaternion_matrix(x[3:7])[:3,:3]
  R = QToRot(x[3:7])
  return p, R

#Convert pos p=(x,y,z) and 3x3 rotation matrix R to a pose, x,y,z,quaternion(qx,qy,qz,qw)
def PosRotToX(p,R):
  M = _rostf_identity_matrix()
  M[:3,:3] = R
  x = list(p)+[0.0]*4
  x[3:7] = _rostf_quaternion_from_matrix(M)
  return x

#Compute "x2 * x1"; x* are [x,y,z,quaternion] form
#x1 can also be [x,y,z] or [quaternion]
#x2 can also be [x,y,z] or [quaternion]
def Transform(x2, x1):
  if len(x2)==3:
    if len(x1)==7:
      x3= [0.0]*7
      x3[:3]= Vec(x2)+Vec(x1[:3])
      x3[3:]= x1[3:]
      return x3
    if len(x1)==3:  #i.e. [x,y,z]
      return Vec(x2)+Vec(x1)
    if len(x1)==4:  #i.e. [quaternion]
      raise Exception('invalid Transform: point * quaternion')
    raise Exception('invalid Transform: For x2 size ({}), invalid x1 size ({}) '.format(len(x2), len(x1)))

  if len(x2)==7:
    p2,R2= XToPosRot(x2)
  elif len(x2)==4:  #i.e. [quaternion]
    p2= Vec([0.0,0.0,0.0])
    R2= QToRot(x2)
  else:
    raise Exception('invalid Transform: Invalid x2 and x1 sizes ({}, {}) '.format(len(x2), len(x1)))

  if len(x1)==7:
    p1,R1= XToPosRot(x1)
    p= np.dot(R2,p1)+p2
    R= np.dot(R2, R1)
    return PosRotToX(p,R)
  if len(x1)==3:  #i.e. [x,y,z]
    p1= x1
    p= np.dot(R2,p1)+p2
    return p
  if len(x1)==4:  #i.e. [quaternion]
    R1= QToRot(x1)
    R= np.dot(R2, R1)
    return RotToQ(R)


#================================

def OptimizeRSPose(sample_list, pos_loss_gain=30.0):
  import scipy.optimize
  def rs_pose_to_x(pose_rs):
    x_rs= list(pose_rs[:3]) + list(RotToQ(Rodrigues(pose_rs[3:])))
    return x_rs
  def loss_diff_x(diff_x):
    pos_err= np.linalg.norm(diff_x[:3])**2
    rot_err= np.linalg.norm(diff_x[3:])**2
    err= pos_loss_gain*pos_err+rot_err
    #print pos_err, rot_err
    #print '{:.2e}'.format(err),
    return err
  num_f_eval= [0]
  def pose_error(pose_rs):
    x_rs= rs_pose_to_x(pose_rs)
    err= sum(loss_diff_x(DiffX(x_marker_robot,Transform(x_rs,x_marker_rs)))
             for (x_marker_robot,x_marker_rs) in sample_list)
    if num_f_eval[0]%100==0:
      sys.stderr.write(' {:.2e}'.format(err))
      sys.stderr.flush()
    num_f_eval[0]+= 1
    return err

  print '##OptimizeRSPose##'
  #print 'sample_list [(x_marker_robot,x_marker_rs)]:'
  #for (x_marker_robot,x_marker_rs) in sample_list:  print ' ',(x_marker_robot,x_marker_rs)
  # Minimize the pose_error
  xmin,xmax= [-5,-5,-5, -5,-5,-5],[5,5,5, 5,5,5]
  tol= 1.0e-6
  bounds= np.array([xmin,xmax]).T
  print 'Optimizing...'
  res= scipy.optimize.differential_evolution(pose_error, bounds, strategy='best1bin', maxiter=300, popsize=20, tol=tol, mutation=(0.5, 1), recombination=0.7)
  #res= scipy.optimize.minimize(pose_error, [0,0,0,0.1,0.1,0.1], bounds=bounds)
  print ''
  print 'Optimization result:\n',res
  x_rs= rs_pose_to_x(res.x)
  print 'Error detail:'
  for i_sample,(x_marker_robot,x_marker_rs) in enumerate(sample_list):
    print '  #{}: err={} diff_x={}'.format(i_sample,
      loss_diff_x(DiffX(x_marker_robot,Transform(x_rs,x_marker_rs))),
      DiffX(x_marker_robot,Transform(x_rs,x_marker_rs)) )
  return x_rs


if __name__=='__main__':
  sample_list= [
      [[0.28015174223103956, -0.23033201855417221, 0.48143342710684306, 0.18290687093012306, -0.49841506824293003, 0.80030691375346885, -0.27863298461119917], [0.27643770043186255, 0.13086763896241135, 0.59451002573089562, -0.36453928406262609, 0.76322472193480873, 0.48100747754320078, -0.23071831472635487]],
      [[0.29381624671589074, -0.24762286620758703, 0.43394628509226868, 0.148868890572323, -0.39693191079019235, 0.85565695464658797, -0.29687419487240568], [0.25915473027886704, 0.14507594510951771, 0.64077353054408781, -0.38866699892868128, 0.81501201803793488, 0.38802647539714935, -0.18473989495604612]],
      [[0.18776265244250817, -0.31469779648728835, 0.42148473937476649, 0.032201405966009851, -0.18721648599377255, 0.92953670221138596, -0.31602939114572376], [0.19349072347291524, 0.040422128524743084, 0.66567943305383637, -0.42856582803438037, 0.88204045383499396, 0.15608960130696423, -0.11820323685752623]],
      [[0.14434927717904017, -0.38347641920692438, 0.37223104009046099, -0.023388115083132224, -0.18957981247884953, 0.98130709589635168, -0.023428066860305775], [0.12451156683941132, -0.0031219107099030672, 0.71759424129253602, -0.67197714392423868, 0.71336714284353275, 0.11521716508082568, -0.16210811955425133]],
      [[0.024361098076881145, -0.39268679104293441, 0.37727327809290218, -0.034605670474585536, -0.1878466012002887, 0.98096727031285424, 0.034918713656888575], [0.11358301211585092, -0.12236084110521474, 0.70722824179181354, 0.7134183875514355, -0.67042047166263496, -0.11374586513517451, 0.16921132835543293]],
      [[0.078366681632210541, -0.44111083940702578, 0.3990501904337217, -0.040028462667675138, -0.019398729769757726, 0.99860165309611659, 0.0285683372544774], [0.066081331597342072, -0.069449930473168706, 0.69271038801799645, 0.72234873504021802, -0.68981977856797283, 0.011380597512979519, 0.047238332765640718]],
      [[0.16558176516246464, -0.42961540821024169, 0.4107014694698361, -0.037948217408960525, 0.098838911647666075, 0.99407621229546617, 0.024561890971301143], [0.078564414151250039, 0.018044799649135971, 0.67921381114774837, 0.71539260975630647, -0.69013917595964758, 0.10343624038431343, -0.034961062440351284]],
      [[0.16027636935000783, -0.5212955753598506, 0.3026494709115688, 0.006909620122392531, 0.067820064325959467, 0.99224925216754745, 0.10389474287724833], [-0.013508773622703025, 0.015175158061249748, 0.78684754358224007, 0.77086294621809404, -0.63283842013070013, 0.058747034007033912, -0.042832676192544196]],
      [[0.24237759840168696, -0.49863534522792158, 0.29879086728266124, -0.026500073238312782, -0.01448355506538905, 0.99862762672045291, -0.042788268271891437], [0.0095480773148380604, 0.098102682024587495, 0.79249215187889333, -0.67160382408636976, 0.73994870861709638, -0.01523105854336371, -0.034528639476591218]],
      [[0.33487294361809172, -0.48926031618983112, 0.30231908166598592, -0.018524562985551482, 0.037525118331173592, 0.99650585176234252, -0.072282732819355128], [0.020018841384360879, 0.19087445682320031, 0.78311996497074932, -0.64726696643879666, 0.7578370267104777, -0.081691774826800803, -0.007414110154757428]],
      [[0.34215608707672823, -0.32794582076784945, 0.28823876117640573, 0.015961232482469962, -0.059599753839404052, 0.98536812974388821, -0.15887969437582256], [0.18117374638577916, 0.19678241359113285, 0.79300782244827217, -0.57881673321881366, 0.81194137222276996, 0.047459593139145891, -0.058906573809488647]],
      [[0.34047562628411521, -0.23385831730529219, 0.28822970164585493, 0.009366709378791388, -0.060987792058421308, 0.99684797100480405, -0.049868594113505284], [0.27497028795264244, 0.19341837348990057, 0.78760389488048821, -0.66153730982980263, 0.74331022285852533, 0.058233906596667849, -0.080418358722669753]],
      [[0.37099699092523958, -0.249243047422992, 0.23503795375035991, 0.0035011375445779609, -0.061595047560637474, 0.99706269108942192, 0.045384823337949944], [0.25921446000088538, 0.22440102241888821, 0.83759448116095536, 0.72783591093190303, -0.67582769980497204, -0.069836275329863584, 0.092923094988809907]],
      [[0.33149992429206571, -0.23533813888841248, 0.28567149689912408, -0.015622482210134202, 0.10005055252318085, 0.99394445424032751, 0.042664351328688663], [0.27485278801020374, 0.18516124280657104, 0.79444779666778254, 0.72987337803830077, -0.6760761838189957, 0.094835144920908582, -0.034815815250161225]],
      [[0.33150559297637139, -0.30614579512505302, 0.28567266222527066, -0.015633201218651235, 0.10004114911482863, 0.99394463565994273, 0.04267824675716695], [0.20409469289740695, 0.18580389370814618, 0.79630568744158636, 0.72927001895971288, -0.67615268177778876, 0.098244267133997545, -0.036480876449283529]],
      [[0.26040566191174236, -0.38124494654865376, 0.27909388285196252, -0.030573834308995938, 0.012677381099930303, 0.99846745233584755, 0.04435393206982953], [0.12790530761442076, 0.1148990873355696, 0.81028414632041146, 0.73329137745357131, -0.67823272594545747, 0.040917612058794441, 0.024695631845410879]],
      [[0.14473469720849835, -0.4102008017047431, 0.26894605755284812, 0.0051088105064538026, -0.030458280622005427, 0.99951672340286712, 0.0035373485489794375], [0.097805450966434143, -0.0012051231146400063, 0.82758142403406565, -0.70517192869644674, 0.70827036352030903, 0.029324996960167386, -0.015022905532197024]],
    ]
  OptimizeRSPose(sample_list)
  #OptimizeRSPose(sample_list, pos_loss_gain=1.0)
  #OptimizeRSPose(sample_list, pos_loss_gain=30.0)
  #OptimizeRSPose(sample_list, pos_loss_gain=3000.0)
