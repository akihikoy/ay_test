//-------------------------------------------------------------------------------------------
/*! \file    kalman_filter.h
    \brief   Kalman filter code (header)
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Oct.15, 2013
*/
//-------------------------------------------------------------------------------------------
#ifndef kalman_filter_h
#define kalman_filter_h
//-------------------------------------------------------------------------------------------
#include <lora/octave.h>
#include <lora/type_gen_oct.h>
#include <boost/function.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
//-------------------------------------------------------------------------------------------

//===========================================================================================
/*!\brief Kalman filter for a linear model;
  state transition model: x_{t+1} = A x_t + B u_t + N(ex;0,R),
  observation model: z_t = C x_t + N(ez;0,Q),
  where:
    x_t: state vector at time t,
    u_t: control vector at time t,
    z_t: observation vector at time t,
    A: matrix,
    B: matrix,
    C: matrix,
    N(x;mu,S): Gaussian probability, x: instance, mu: mean, S: covariance matrix,
    ex: state transition noise,
    R: covariance matrix of ex,
    ez: observation noise,
    Q: covariance matrix of ez.
*/
class TKalmanFilter
//===========================================================================================
{
public:
  typedef ColumnVector TVector;
  typedef Matrix TMatrix;

  TKalmanFilter() : tol_(1.0e-6) {}
  ~TKalmanFilter() {}

  //! Provide SetA, GetA, etc.
  #define DEF_SET_GET(x_mat) \
    void Set##x_mat(const TMatrix &val) {x_mat##_= val;}  \
    const TMatrix& Get##x_mat() const {return x_mat##_;}
  DEF_SET_GET(A)
  DEF_SET_GET(B)
  DEF_SET_GET(C)
  DEF_SET_GET(R)
  DEF_SET_GET(Q)
  #undef DEF_SET_GET

  //! Get the current estimation of the mean
  const TVector& GetMu() const {return mu_;}

  //! Get the current estimation of the covariance matrix
  const TMatrix& GetSigma() const {return Sigma_;}

  //! Initialize the estimation
  virtual void Initialize(const TVector &mu, const TMatrix &Sigma)
    {
      mu_= mu;
      Sigma_= Sigma;
      eye_= GetEye(GenSize(mu_));
    }

  //! Update the estimation (mu_, Sigma_) from a control u an observation z
  virtual void Update(const TVector &u, const TVector &z)
    {
      // estimation:
      TVector mu_est;
      if(GenSize(u)!=0)  mu_est= A_ * mu_ + B_ * u;
      else  mu_est= A_ * mu_;
      // kalman gain:
      TMatrix K= do_internal_proc();
      // update:
      mu_= mu_est + K * (z - C_ * mu_est);
    }

protected:
  TMatrix A_, B_, C_, R_, Q_;
  TVector mu_;
  TMatrix Sigma_;
  TMatrix eye_;  // unit matrix whose size corresponds with mu_
  double tol_;  // tolerarnce of pseudo_inverse

  TMatrix do_internal_proc()
    {
      // estimation:
      TMatrix Sigma_est= A_ * Sigma_ * A_.transpose() + R_;
      // kalman gain:
      TMatrix K= Sigma_est * C_.transpose() * (C_ * Sigma_est * C_.transpose() + Q_).pseudo_inverse(tol_);
      // update:
      Sigma_= (eye_ - K * C_) * Sigma_est;
      return K;
    }

};
//-------------------------------------------------------------------------------------------

//===========================================================================================
/*!\brief Extended kalman filter (EKF) for a general model;
  state transition model: x_{t+1} = g(x_t, u_t) + N(ex;0,R),
  observation model: z_t = h(x_t) + N(ez;0,Q),
  where:
    x_t: state vector at time t,
    u_t: control vector at time t,
    z_t: observation vector at time t,
    g(x,u): state transition model,
    h(x): observation model, x: state,
    N(x;mu,S): Gaussian probability, x: instance, mu: mean, S: covariance matrix,
    ex: state transition noise,
    R: covariance matrix of ex,
    ez: observation noise,
    Q: covariance matrix of ez.
*/
class TExtendedKalmanFilter : public TKalmanFilter
//===========================================================================================
{
public:
  //! Set a state transition model g
  void SetStateTransModel(const boost::function<TVector(const TVector &x, const TVector &u)> &f)  {state_trans_model_= f;}
  //! Set an observation model h
  void SetObservationModel(const boost::function<TVector(const TVector &x)> &f)  {obs_model_= f;}

  //! Set a constraining-state function
  void SetStateConstraint(const boost::function<void(TVector &x)> &f)  {constrain_state_= f;}

  //! Set a derivative model of g w.r.t. a state vector
  void SetG(const boost::function<TMatrix(const TVector &x, const TVector &u)> &f)  {func_G_= f;}
  //! Set a derivative model of h w.r.t. a state vector
  void SetH(const boost::function<TMatrix(const TVector &x)> &f)  {func_H_= f;}

  //! Update the estimation (mu_, Sigma_) from a control u an observation z
  virtual void Update(const TVector &u, const TVector &z)
    {
      // estimation:
      TVector mu_est= state_trans_model_(mu_,u);
      A_= func_G_(mu_,u);
      C_= func_H_(mu_est);
      // kalman gain:
      TMatrix K= do_internal_proc();
      // update:
LDBGVAR(K);
LDBGVAR((z - obs_model_(mu_est)).transpose());
LDBGVAR((K * (z - obs_model_(mu_est))).transpose());
      mu_= mu_est + K * (z - obs_model_(mu_est));
      if(constrain_state_)  constrain_state_(mu_);
    }

protected:
  //! Hide (make inaccessible) SetA, GetA, etc.
  #define HIDE_SET_GET(x_mat) \
    void Set##x_mat(const TMatrix &val);  \
    const TMatrix& Get##x_mat() const;
  HIDE_SET_GET(A)
  HIDE_SET_GET(B)
  HIDE_SET_GET(C)
  #undef DEF_SET_GET

  boost::function<TVector(const TVector &x, const TVector &u)>  state_trans_model_;
  boost::function<TVector(const TVector &x)>  obs_model_;

  boost::function<void(TVector &x)>  constrain_state_;

  boost::function<TMatrix(const TVector &x, const TVector &u)>  func_G_;
  boost::function<TMatrix(const TVector &x)>  func_H_;

};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of loco_rabbits
//-------------------------------------------------------------------------------------------
#endif // kalman_filter_h
//-------------------------------------------------------------------------------------------
