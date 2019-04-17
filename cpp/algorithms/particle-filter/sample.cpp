//-------------------------------------------------------------------------------------------
/*! \file    sample.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.09, 2012

    \note
      monitor with:
        qplot -s 'set xrange [-1:7];set yrange [-0.5:4.5]' particles.dat u 1:2 obsv.dat u 2:3 state.dat u 2:3 est.dat u 2:3 w l lw 3 -i 0.1
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
#include <lora/math.h>
// #include <lora/file.h>
#include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <iostream>
#include <fstream>
// #include <iomanip>
// #include <string>
#include <vector>
#include <unistd.h>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

const double DT(0.01);
const double WEIGHT_SIGMA(0.5);
const double TRANS_NOISE(0.05);
const double OBSERV_NOISE(0.5);

struct TParticle
{
  double X, Y;
  double Vx, Vy;
  TParticle() : X(0.0), Y(0.0), Vx(0.0), Vy(0.0) {}
};

struct TParticleW
{
  TParticle P;
  double W;
  double prev_sum_w_;  // temporary variable
  TParticleW() : P(), W(0.0), prev_sum_w_(0.0) {}
};

struct TObservation
{
  double X, Y;
};

TParticle operator+(const TParticle &lhs, const TParticle &rhs)
{
  TParticle res(lhs);
  res.X+=  rhs.X;
  res.Y+=  rhs.Y;
  res.Vx+= rhs.Vx;
  res.Vy+= rhs.Vy;
  return res;
}

TParticle operator*(const TParticle &lhs, const double &rhs)
{
  TParticle res(lhs);
  res.X*=  rhs;
  res.Y*=  rhs;
  res.Vx*= rhs;
  res.Vy*= rhs;
  return res;
}

std::ostream& operator<<(std::ostream &lhs, const TParticle &rhs)
{
  lhs<<rhs.X<<" "<<rhs.Y<<" "<<rhs.Vx<<" "<<rhs.Vy;
  return lhs;
}

std::ostream& operator<<(std::ostream &lhs, const TParticleW &rhs)
{
  lhs<<rhs.P<<" "<<rhs.W;
  return lhs;
}

std::ostream& operator<<(std::ostream &lhs, const TObservation &rhs)
{
  lhs<<rhs.X<<" "<<rhs.Y;
  return lhs;
}

void GenerateParticles(std::vector<TParticleW> &particles, int N)
{
  particles.resize(N);
  double w(1.0/static_cast<double>(N));
  for(std::vector<TParticleW>::iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
  {
    itr->P.X= Rand(-1.0,1.0);
    itr->P.Y= Rand(-1.0,1.0);
    itr->P.Vx= Rand(-0.1,0.1);
    itr->P.Vy= Rand(-0.1,0.1);
    itr->W= w;
  }
}

TParticle AverageParticles(const std::vector<TParticleW> &particles)
{
  TParticle p;
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    p= p + itr->P * itr->W;
  return p;
}

void PrintParticles(std::ostream &os, const std::vector<TParticleW> &particles)
{
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    os<<*itr<<std::endl;
}

double SumWeights(std::vector<TParticleW> &particles)
{
  double sumw(0.0);
  for(std::vector<TParticleW>::iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    sumw+= itr->W;
  return sumw;
}

void NormalizeWeights(std::vector<TParticleW> &particles)
{
  double sumw(SumWeights(particles));
  for(std::vector<TParticleW>::iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    itr->W/= sumw;
}

void ComputePrevSumW(std::vector<TParticleW> &particles)
{
  double psumw(0.0);
  for(std::vector<TParticleW>::iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
  {
    psumw+= itr->W;
    itr->prev_sum_w_= psumw;
  }
}

void TransitionModel(const TParticle &curr, TParticle &next, bool add_noise=true)
{
  next.X= curr.X + DT*curr.Vx + (add_noise ? (TRANS_NOISE*NDRand()) : 0.0);
  next.Y= curr.Y + DT*curr.Vy + (add_noise ? (TRANS_NOISE*NDRand()) : 0.0);
  next.Vx= curr.Vx + (add_noise ? (TRANS_NOISE*NDRand()) : 0.0);
  next.Vy= curr.Vy + (add_noise ? (TRANS_NOISE*NDRand()) : 0.0);
}

double ComputeWeight(const TParticle &p, const TObservation &o)
{
  double w= real_exp(-0.5*WEIGHT_SIGMA*(Square(o.X-p.X)+Square(o.Y-p.Y)));
  return w;
}

template <typename PFwdItr>
inline int SelectFromProbVec (PFwdItr pfirst, PFwdItr plast)
{
  int index(0);
  TReal p (Rand(1.0l));
  for (; pfirst!=plast; ++pfirst, ++index)
  {
    if (p<=pfirst->W)  return index;
    p -= pfirst->W;
  }
  return index-1;
}

int SelectFromPrevSumW (const std::vector<TParticleW> &particles)
{
  TReal p (Rand(1.0l));
  int imin(0), imax(particles.size()-1), index((imin+imax)/2);

  while(true)
  {
    if(p > particles[index].prev_sum_w_)
    {
      imin= index;
      if(index >= imax)  return imax;  // index==imax
      else if(index == imax-1)  ++index;
      else  index= (index+imax)/2;
    }
    else if(index<=imin)  // index==imin
    {
      return index;
    }
    else if(p > particles[index-1].prev_sum_w_)
    {
      return index;
    }
    else  // p <= particles[index-1].prev_sum_w_
    {
      imax= index;
      index= (imin+index)/2;
    }
  }
}

void UpdateParticles(std::vector<TParticleW> &particles, const TObservation &o)
{
  std::vector<TParticleW> pnew(particles.size());
  std::vector<TParticleW>::iterator nitr(pnew.begin());
  for(std::vector<TParticleW>::iterator pitr(particles.begin()),plast(particles.end());pitr!=plast;++pitr,++nitr)
  {
    TransitionModel(pitr->P,nitr->P);
    nitr->W= ComputeWeight(nitr->P,o);
  }
  NormalizeWeights(pnew);
  ComputePrevSumW(pnew);
  for(std::vector<TParticleW>::iterator pitr(particles.begin()),plast(particles.end());pitr!=plast;++pitr)
  {
    // int i= SelectFromProbVec(pnew.begin(),pnew.end());
    int i= SelectFromPrevSumW(pnew);
    *pitr= pnew[i];
  }
  NormalizeWeights(particles);
}

void UpdateParticles(std::vector<TParticleW> &particles)
{
  TParticle next;
  for(std::vector<TParticleW>::iterator pitr(particles.begin()),plast(particles.end());pitr!=plast;++pitr)
  {
    TransitionModel(pitr->P,next,false);
    pitr->P= next;
  }
}


int main(int argc, char**argv)
{
  unsigned seed ((unsigned)time(NULL));
  XSrand(seed);

  ofstream ofs_state("res/state.dat"), ofs_obsv("res/obsv.dat"), ofs_est("res/est.dat");
  std::vector<TParticleW>  particles;

  GenerateParticles(particles, 2000);

  double x(0.0), y(0.0);
  double vx(0.0), vy(0.0);
  double dt(0.01);
  bool observable(true);
  for(double t(0.0);t<10.0;t+=dt)
  {
    if(t<3.0)        {vx=0.0; vy=0.0;}
    else if(t<6.0)   {vx=1.0; vy=0.0;}
    else if(t<9.0)   {vx=1.0; vy=1.0;}
    else             {vx=0.0; vy=0.0;}
    if(t<5.0)        {observable=true;}
    else if(t<7.0)   {observable=false;}
    else             {observable=true;}
    x+= vx*dt;
    y+= vy*dt;

    TObservation o;
    if(observable)
    {
      o.X= x + OBSERV_NOISE*NDRand();
      o.Y= y + OBSERV_NOISE*NDRand();
      UpdateParticles(particles, o);
    }
    else
    {
      UpdateParticles(particles);
    }

    cerr<<SumWeights(particles)<<endl;
    TParticle est= AverageParticles(particles);

    if(observable)
    {
      ofs_state<<t<<" "<<x<<" "<<y<<endl;
      ofs_obsv<<t<<" "<<o<<endl;
    }
    else
    {
      ofs_state<<endl;
      ofs_obsv<<endl;
    }
    ofs_est<<t<<" "<<est<<endl;

    cout<<t<<" "<<est<<endl;

    #if 1
    ofstream ofs_part("res/particles.dat");
    PrintParticles(ofs_part, particles);
    usleep(10000);
    #endif
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
