//-------------------------------------------------------------------------------------------
/*! \file    dmp1.h
    \brief   DMP test code
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.26, 2013
*/
//-------------------------------------------------------------------------------------------
#ifndef dmp1_h
#define dmp1_h
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cmath>
#include <valarray>
#include <numeric>
#include <Eigen/Dense>
//-------------------------------------------------------------------------------------------
#include <fstream>  // for debug
//-------------------------------------------------------------------------------------------
#ifndef LDBGVAR
#define LDBGVAR(var) std::cout<<#var"= "<<(var)<<std::endl
#endif
//-------------------------------------------------------------------------------------------
namespace movement_primitives
{
//-------------------------------------------------------------------------------------------

#if 0
template <typename t_value>
class TDynamicMovementPrimitives
{
public:
  typedef t_value  TValue;

  TDynamicMovementPrimitives()  {}

  void SetDoFs(int n)
    {
      y_.resize(n); dy_.resize(n);
      z_.resize(n); dz_.resize(n);
      goal_.resize(n); start_.resize(n);
      alpha_z_.resize(n); beta_z_.resize(n);
    }

  void Step(const TValue &dt)
    {

    }

protected:
  TValue  x_, dx_;  // state of the canonical system
  std::valarray<TValue>  y_, dy_;  // output
  std::valarray<TValue>  z_, dz_;
  std::valarray<TValue>  goal_, start_;

  std::valarray<TValue>  alpha_z_, beta_z_;
  TValue alpha_x_, tau_;

  std::valarray<TValue>  c_, h_;  // kernel centers, band width
  std::valarray<TValue>  w_;  // weight parameters ([w of 0-th kernel], [c of 1-st kernel], ..)

};
#endif

template <typename t_value>
inline t_value Square(const t_value &x)
{
  return x*x;
}

template <typename t_value>
class TDynamicMovementPrimitives
{
public:
  typedef t_value  TValue;

  TDynamicMovementPrimitives()  {}

  void SetAlphaZ(const TValue &v) {alpha_z_= v;}
  void SetBetaZ(const TValue &v) {beta_z_= v;}
  void SetAlphaX(const TValue &v) {alpha_x_= v;}

  void Init()
    {
      x_= 1.0l;
      y_= start_;
      z_= tau_*start_v_;
dy_= 0.0l;
      t_= 0.0l;
      active_= true;
    }

  void Step(const TValue &dt, const TValue *curr_y=NULL, const TValue *curr_v=NULL)
    {
      std::valarray<TValue>  phi(c_.size());
      get_phi(x_, phi);
      TValue f= std::inner_product(&phi[0],&phi[0]+phi.size(),&w_[0],0.0l)*(goal_-start_);

// double dy_old(dy_);
      dz_= (alpha_z_*(beta_z_*(goal_-y_)-z_)+f) / tau_;
      dy_= z_ / tau_;
      dx_= -alpha_x_*x_ / tau_;
if(curr_y)
{
// LDEBUG(Square((*curr_y-y_)/dt));
  // dx_*= 1.0l/(1.0l+0.005*Square((*curr_y-y_)/dt));
  dx_*= 1.0l/(1.0l+0.2*std::fabs((*curr_y-y_)/dt));
  // LDEBUG(std::fabs((*curr_y-y_)/dy_old));
  // if(std::fabs(dy_old)>1.0e-6)
    // dx_*= std::fabs((*curr_y-y_)/dy_old);
  y_= *curr_y;
  if(curr_v)
    z_= tau_*(*curr_v);
}
if(t_>0.1&&t_<0.9)
{
// dx_=0.0;
// dy_=0.0;
// z_=0.0;
}
      z_+= dt*dz_;
      y_+= dt*dy_;
      x_+= dt*dx_;
      t_+= dt;
if(active_&&x_<0.05)  // FIXME: the parameter
{
  std::cerr<<t_<<" [s]: canonical dynamics: 99%"<<std::endl;
  active_= false;
}

    }

  const TValue& Y()  {return y_;}

  void LearnFromDemo(const std::valarray<TValue> demo,  const TValue &T, int num_basis)
    {
      TValue dt= T/static_cast<TValue>(demo.size());
      tau_= -alpha_x_*T / std::log(0.05); // FIXME replace by a parameter variable
// tau_=0.8;
      start_= demo[0];
      goal_= demo[demo.size()-1];
      // FIXME for too small goal_-start_
start_v_= (demo[1]-demo[0])/dt;

      c_.resize(num_basis);
      TValue xw= 1.0l/static_cast<TValue>(num_basis-1);
      c_[0]= 0.0;
std::ofstream ofs_ftrg("res/f_trg.dat");
      for(int i(1); i<num_basis; ++i)  c_[i]= c_[i-1]+xw;
// for(int i(0); i<num_basis; ++i) c_[i]= std::exp(-alpha_x_/tau_*((double)i/(double)num_basis));
      h_.resize(num_basis);
      for(int i(0); i<num_basis; ++i)  h_[i]= Square(1.0/xw)*4.0;
      w_.resize(num_basis);

      using namespace Eigen;
      MatrixXd Phi(MatrixXd::Zero(demo.size(), num_basis));
      VectorXd F(VectorXd::Zero(demo.size()));
      x_= 1.0l;
      std::valarray<TValue>  phi(num_basis);
      dy_= 0.0;
      TValue ddy(0.0);
      for(int n(0),n_last(demo.size()); n<n_last; ++n)
      {
        get_phi(x_, phi);
        phi*= (goal_-start_);
        for(int r(0),r_last(phi.size()); r<r_last; ++r)  Phi(n,r)= phi[r];

        y_= demo[n];
        if(n>=1)  dy_= (demo[n]-demo[n-1])/dt;  else dy_= 0.0;
        if(n>=2)  ddy= (demo[n]-2.0*demo[n-1]+demo[n-2])/Square(dt);  else ddy= 0.0;
        F(n)= Square(tau_)*ddy -alpha_z_*(beta_z_*(goal_-y_)-tau_*dy_);
ofs_ftrg<<x_<<" "<<F(n)<<std::endl;

        dx_= -alpha_x_*x_ / tau_;
        x_+= dt*dx_;
      }

      const TValue lambda(0.001);  // regularization param
      Map<VectorXd> w(&w_[0],num_basis);
      w= (Phi.transpose()*Phi+lambda*MatrixXd::Identity(num_basis,num_basis)).inverse()*Phi.transpose() * F;
// std::cout<<F<<std::endl<<std::endl;
// std::cout<<Phi*w<<std::endl;

// LDBGVAR(Phi);
// LDBGVAR(F.transpose());
// LDBGVAR(w.transpose());
    }

// protected:
  TValue  x_, dx_;  // state of the canonical system
  TValue  y_, dy_;  // output
  TValue  z_, dz_;
  TValue  goal_, start_, start_v_;

  TValue  alpha_z_, beta_z_;
  TValue alpha_x_;
  TValue tau_;
TValue t_;
bool active_;

  std::valarray<TValue>  c_, h_;  // kernel centers, band width
  std::valarray<TValue>  w_;  // weight parameters ([w of 0-th kernel], [c of 1-st kernel], ..)

  void get_phi(const TValue &x, std::valarray<TValue> &phi) const
    {
      phi.resize(c_.size());
      TValue sum(0.0l), ex;
      for(int i(0),N(phi.size()); i<N; ++i)
      {
        ex= std::exp(-h_[i]*Square(x-c_[i]));
        phi[i]= x*ex;
        sum+= ex;
      }
      phi/= sum;
    }

};


//-------------------------------------------------------------------------------------------
}  // end of movement_primitives
//-------------------------------------------------------------------------------------------
#endif // dmp1_h
//-------------------------------------------------------------------------------------------
