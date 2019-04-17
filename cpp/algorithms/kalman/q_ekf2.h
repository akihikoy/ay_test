//-------------------------------------------------------------------------------------------
/*! \file    q_ekf.h.h
    \brief   EKF with quaternion; separated version (header)
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.31, 2013

    qplot -3d -s 'set size 1,1.2;set ticslevel 0;set xrange [-1:1];set yrange [-1:1];set ange[-1:1]' res/box.dat w l -i 0.1
*/
//-------------------------------------------------------------------------------------------
#ifndef q_ekf_h_h
#define q_ekf_h_h
//-------------------------------------------------------------------------------------------
#include "kalman_filter.h"
#include <lora/stl_math.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------

//===========================================================================================
/*!\brief Model functions of IMU with accelerometer and gyro for EKF: separated version;
    state: x_q=(q_w,q_x,q_y,q_z, w_x,w_y,w_z),
    state: x_p=(x,y,z, v_x,v_y,v_z, a_x,a_y,a_z),
    where:
      (q_w,q_x,q_y,q_z): quaternion in the world frame [0,1,2,3],
      (w_x,w_y,w_z): angular velocity in the world frame [4,5,6],
      (x,y,z): position in the world frame [0,1,2],
      (v_x,v_y,v_z): velocity in the world frame [3,4,5],
      (a_x,a_y,a_z): acceleration in the world frame [6,7,8],
    control: u_p=(), u_q=()
    observation: z_q=(w_x',w_y',w_z'),
    observation: z_p=(a_x,a_y,a_z),
    where:
      (w_x',w_y',w_z'): angular velocity in the local (sensor) frame [0,1,2],
      (a_x,a_y,a_z): acceleration in the world frame [0,1,2],
      (a_x',a_y',a_z'): acceleration in the local (sensor) frame [not used],
*/
//===========================================================================================

//-------------------------------------------------------------------------------------------
namespace q_ekf
{
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
namespace detail
{
//-------------------------------------------------------------------------------------------
typedef TKalmanFilter::TVector TVector;
typedef TKalmanFilter::TMatrix TMatrix;
static const int IDX_Q(0);
static const int IDX_W(4);
static const int IDX_P(0);
static const int IDX_V(3);
static const int IDX_A(6);
static const int IDX_ZW(0);
static const int IDX_ZA(0);
inline TVector EXT_Q(const TVector &x)  {return x.extract_n(IDX_Q,4);}
inline TVector EXT_W(const TVector &x)  {return x.extract_n(IDX_W,3);}
inline TVector EXT_P(const TVector &x)  {return x.extract_n(IDX_P,3);}
inline TVector EXT_V(const TVector &x)  {return x.extract_n(IDX_V,3);}
inline TVector EXT_A(const TVector &x)  {return x.extract_n(IDX_A,3);}
inline TVector EXT_ZW(const TVector &z)  {return z.extract_n(IDX_ZW,3);}
inline TVector EXT_ZA(const TVector &z)  {return z.extract_n(IDX_ZA,3);}
inline void ASIGN_P(TVector &x,const TVector &p)  {x.insert(p,IDX_P);}
inline void ASIGN_W(TVector &x,const TVector &w)  {x.insert(w,IDX_W);}
inline void ASIGN_Q(TVector &x,const TVector &q)  {x.insert(q,IDX_Q);}
inline void ASIGN_V(TVector &x,const TVector &v)  {x.insert(v,IDX_V);}
inline void ASIGN_A(TVector &x,const TVector &a)  {x.insert(a,IDX_A);}
inline void ASIGN_ZW(TVector &z,const TVector &w)  {z.insert(w,IDX_ZW);}
inline void ASIGN_ZA(TVector &z,const TVector &a)  {z.insert(a,IDX_ZA);}

// GetWQMat()*q: rotate a quaternion q with an ang-vel w in dt
inline TMatrix GetWQMat(const TVector &w, const TReal &dt, const TReal &tol=1.0e-6)
{
  TReal norm= GetNorm(w);
  if(norm<tol)  return GetEye(4);
  TReal c= real_cos(dt*norm*0.5l), s_n= real_sin(dt*norm*0.5l)/norm;
  TReal sn1= s_n*w(0), sn2= s_n*w(1), sn3= s_n*w(2);
  TMatrix M(4,4);
  M(0,0)=   c; M(0,1)= -sn1; M(0,2)= -sn2; M(0,3)= -sn3;
  M(1,0)= sn1; M(1,1)=    c; M(1,2)= -sn3; M(1,3)=  sn2;
  M(2,0)= sn2; M(2,1)=  sn3; M(2,2)=    c; M(2,3)= -sn1;
  M(3,0)= sn3; M(3,1)= -sn2; M(3,2)=  sn1; M(3,3)=    c;
  return M;
}
// Get a derivative of a quaternion q w.r.t. an ang-vel w (approximation version)
inline TMatrix GetDQDWApprox(const TVector &q, const TReal &dt)
{
  TMatrix W(4,3);
  const TypeExt<TVector>::value_type &qw(q(0)),&qx(q(1)),&qy(q(2)),&qz(q(3));
  W(0,0)= -qx; W(0,1)= -qy; W(0,2)= -qz;
  W(1,0)=  qw; W(1,1)=  qz; W(1,2)= -qy;
  W(2,0)= -qz; W(2,1)=  qw; W(2,2)=  qx;
  W(3,0)=  qy; W(3,1)= -qx; W(3,2)=  qw;
  return W*(dt*0.5l);
}

// convert a quaternion q to the rotation matrix R
inline TMatrix QtoR(const TVector &q)
{
  TMatrix M(3,3);
  const TypeExt<TVector>::value_type &qw(q(0)),&qx(q(1)),&qy(q(2)),&qz(q(3));
  M(0,0)= qw*qw+qx*qx-qy*qy-qz*qz; M(0,1)= 2.0*(qx*qy-qw*qz);       M(0,2)= 2.0*(qx*qz+qw*qy);
  M(1,0)= 2.0*(qx*qy+qw*qz);       M(1,1)= qw*qw-qx*qx+qy*qy-qz*qz; M(1,2)= 2.0*(qy*qz-qw*qx);
  M(2,0)= 2.0*(qx*qz-qw*qy);       M(2,1)= 2.0*(qy*qz+qw*qx);       M(2,2)= qw*qw-qx*qx-qy*qy+qz*qz;
  return M;
}
inline TMatrix InvQtoR(const TVector &q)
{
  return QtoR(q).transpose();
}

inline TMatrix Combine(const TMatrix &W0, const TMatrix &W1, const TMatrix &W2, const TMatrix &W3, const TVector &v)
{
  TMatrix M(GenSize(v),4);
  M.insert(W0*v,0,0); M.insert(W1*v,0,1); M.insert(W2*v,0,2); M.insert(W3*v,0,3);
  return M;
}

static const double DAMPING_FACTOR_QW(1.00);
static const double DAMPING_FACTOR_QX(0.98);
static const double DAMPING_FACTOR_QY(0.98);
static const double DAMPING_FACTOR_QZ(1.00);
static const double DAMPING_FACTOR_V(0.99);

//-------------------------------------------------------------------------------------------
}  // end of detail
//-------------------------------------------------------------------------------------------

void func_constrain_state_q(TKalmanFilter::TVector &x);
void func_constrain_state_p(TKalmanFilter::TVector &x);

TKalmanFilter::TVector func_state_trans_q(const TKalmanFilter::TVector &x, const TKalmanFilter::TVector &u, const TReal &dt)
{
  using namespace detail;
  TMatrix q_damping(4,4,0.0);q_damping(0,0)=DAMPING_FACTOR_QW;q_damping(1,1)=DAMPING_FACTOR_QX;q_damping(2,2)=DAMPING_FACTOR_QY;q_damping(3,3)=DAMPING_FACTOR_QZ;
  TVector next_x(x);
  // ASIGN_Q(next_x, GetWQMat(EXT_W(x),dt)*EXT_Q(x));
  /*DEBUG:damping*/ASIGN_Q(next_x, q_damping*(GetWQMat(EXT_W(x),dt)*EXT_Q(x)));
  ASIGN_W(next_x, EXT_W(x));
  func_constrain_state_q(next_x);
// LDBGVAR(EXT_Q(next_x).transpose());
  return next_x;
}
TKalmanFilter::TVector func_state_trans_p(const TKalmanFilter::TVector &x, const TKalmanFilter::TVector &u, const TReal &dt)
{
  using namespace detail;
  TVector next_x(x);
  ASIGN_P(next_x, EXT_P(x)+dt*EXT_V(x));
  // ASIGN_V(next_x, EXT_V(x)+dt*EXT_A(x));
  /*DEBUG:damping*/ASIGN_V(next_x, DAMPING_FACTOR_V*(EXT_V(x)+dt*EXT_A(x)));
  ASIGN_A(next_x, EXT_A(x));
  func_constrain_state_p(next_x);
// LDBGVAR(EXT_Q(next_x).transpose());
  return next_x;
}
//-------------------------------------------------------------------------------------------

void func_constrain_state_q(TKalmanFilter::TVector &x)
{
  using namespace detail;
  // constrain ||q||=1
  TVector q(EXT_Q(x));
  TReal q_norm(GetNorm(q));
  ASIGN_Q(x, q/q_norm);
}
void func_constrain_state_p(TKalmanFilter::TVector &x)
{
  using namespace detail;
  // constrain z>0
  // if(x(IDX_P+2)<0.0)  x(IDX_P+2)= 0.0;
}
//-------------------------------------------------------------------------------------------

TKalmanFilter::TMatrix func_G_q(const TKalmanFilter::TVector &x, const TKalmanFilter::TVector &u, const TReal &dt)
{
  using namespace detail;
  TMatrix G(7,7,0.0), I3(GetEye(3));
  TMatrix W1(GetWQMat(EXT_W(x),dt)), W2(GetDQDWApprox(EXT_Q(x),dt));

  #define GB(x_r,x_c, x_mat) G.insert(x_mat,IDX_##x_r,IDX_##x_c)
  GB(Q,Q, W1);  GB(Q,W, W2);
  /*GB(W,Q,);*/ GB(W,W,I3);
  #undef GB
  return G;
}
TKalmanFilter::TMatrix func_G_p(const TKalmanFilter::TVector &x, const TKalmanFilter::TVector &u, const TReal &dt)
{
  using namespace detail;
  TMatrix G(9,9,0.0), I3(GetEye(3)), dtI3(dt*I3);
  TMatrix W1(GetWQMat(EXT_W(x),dt)), W2(GetDQDWApprox(EXT_Q(x),dt));

  #define GB(x_r,x_c, x_mat) G.insert(x_mat,IDX_##x_r,IDX_##x_c)
  GB(P,P, I3);  GB(P,V, dtI3); /*GB(P,A,);*/
  /*GB(V,P,);*/ GB(V,V,I3);    GB(V,A,dtI3);
  /*GB(A,P,);*/ /*GB(A,V,);*/  GB(A,A,I3);
  #undef GB
  return G;
}
//-------------------------------------------------------------------------------------------

TKalmanFilter::TVector func_observation_q(const TKalmanFilter::TVector &x, const TReal &dt, const TKalmanFilter::TVector &noise_mean)
{
  using namespace detail;
  TVector z(3);
  TMatrix iR= InvQtoR(EXT_Q(x));
  ASIGN_ZW(z, iR*EXT_W(x) + EXT_ZW(noise_mean));
  return z;
}
TKalmanFilter::TVector func_observation_p(const TKalmanFilter::TVector &x, const TReal &dt, const TKalmanFilter::TVector &g, const TKalmanFilter::TVector &noise_mean)
{
  using namespace detail;
  TVector z(3);
  ASIGN_ZA(z, EXT_A(x)+g + EXT_ZA(noise_mean));
  return z;
}
//-------------------------------------------------------------------------------------------

TKalmanFilter::TMatrix func_H_q(const TKalmanFilter::TVector &x, const TReal &dt)
{
  using namespace detail;
  TMatrix W31(3,3),W32(3,3),W33(3,3),W34(3,3);
  const TypeExt<TVector>::value_type &qw(x(IDX_Q+0)),&qx(x(IDX_Q+1)),&qy(x(IDX_Q+2)),&qz(x(IDX_Q+3));
  W31(0,0)=  qw; W31(0,1)=  qz; W31(0,2)= -qy;
  W31(1,0)= -qz; W31(1,1)=  qw; W31(1,2)=  qx;
  W31(2,0)=  qy; W31(2,1)= -qx; W31(2,2)=  qw;
  W31*= 2.0;
  W32(0,0)=  qx; W32(0,1)=  qy; W32(0,2)=  qz;
  W32(1,0)=  qy; W32(1,1)= -qx; W32(1,2)=  qw;
  W32(2,0)=  qz; W32(2,1)= -qw; W32(2,2)= -qx;
  W32*= 2.0;
  W33(0,0)= -qy; W33(0,1)=  qx; W33(0,2)= -qw;
  W33(1,0)=  qx; W33(1,1)=  qy; W33(1,2)=  qz;
  W33(2,0)=  qw; W33(2,1)=  qz; W33(2,2)= -qy;
  W33*= 2.0;
  W34(0,0)= -qz; W34(0,1)=  qw; W34(0,2)=  qx;
  W34(1,0)= -qw; W34(1,1)= -qz; W34(1,2)=  qy;
  W34(2,0)=  qx; W34(2,1)=  qy; W34(2,2)=  qz;
  W34*= 2.0;
  TMatrix W3;
  W3= Combine(W31,W32,W33,W34,EXT_W(x));
// LDBGVAR(W3);
// LDBGVAR(W4);

  TMatrix H(3,7,0.0);
  TMatrix iR= InvQtoR(EXT_Q(x));
  #define HB(x_r,x_c, x_mat) H.insert(x_mat,IDX_Z##x_r,IDX_##x_c)
  HB(W,Q,W3); HB(W,W,iR);
  #undef HB
LDBGVAR(H);
  return H;
}
TKalmanFilter::TMatrix func_H_p(const TKalmanFilter::TVector &x, const TReal &dt)
{
  using namespace detail;
  TMatrix H(3,9,0.0), I3(GetEye(3));
  #define HB(x_r,x_c, x_mat) H.insert(x_mat,IDX_Z##x_r,IDX_##x_c)
  /*HB(A,P,);*/ /*HB(A,V,);*/ HB(A,A,I3);
  #undef HB
// LDBGVAR(H);
  return H;
}
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
}  // end of q_ekf
//-------------------------------------------------------------------------------------------
}  // end of loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // q_ekf_h_h
//-------------------------------------------------------------------------------------------
