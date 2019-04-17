// marker tracking using a particle filter

#include <cv.h>
#include <highgui.h>

#include <cassert>
#include <iostream>
#include <fstream>

#include <lora/rand.h>
#include <lora/math.h>

#include "edgeldetector.h"
#include "linesegment.h"
#include "buffer.h"

const char  * WINDOW_NAME  = "Marker Tracker";
//const CFIndex CASCADE_NAME_LEN = 2048;

struct TTetragon
{
  Vector2f c[4];
};

inline Vector2f ToVector2f(const cv::Vec<double,2> &v)
{
  return Vector2f(v[0],v[1]);
}

using namespace std;
using namespace loco_rabbits;
cv::Mat  draw_image;

struct debugLine
{
  int x1, y1, x2, y2, r, g, b, t;
};

std::vector< debugLine > debugLines;

void debugDrawAll()
{
  cv::Point start, end;

  for( int i=0, s=debugLines.size(); i<s; i++ )
  {
    start.x = debugLines[i].x1;
    start.y = debugLines[i].y1;

    end.x = debugLines[i].x2;
    end.y = debugLines[i].y2;

    if(std::abs(start.x)+std::abs(start.y) > 10000 || std::abs(end.x)+std::abs(end.y) > 10000)
      continue;
    // cerr<<"line:"<<start.x<<","<<start.y<<" --> "<<end.x<<","<<end.y<<endl;
    cv::line(draw_image, start, end, cv::Scalar(debugLines[i].r,debugLines[i].g,debugLines[i].b), debugLines[i].t);
  }

  debugLines.resize(0);
}

void debugDrawLine(int x1, int y1, int x2, int y2, int r, int g, int b, int t)
{
  debugLine newLine;

  newLine.x1 = x1; newLine.y1 = y1;
  newLine.x2 = x2; newLine.y2 = y2;
  newLine.r = r; newLine.g = g; newLine.b = b;
  newLine.t = t;

  debugLines.push_back(newLine);
}

void debugDrawPoint(int x1, int y1, int r, int g, int b, int t)
{
  debugDrawLine(x1-0, y1-1, x1+0, y1+1, r, g, b, t);
  debugDrawLine(x1-1, y1-0, x1+1, y1-0, r, g, b, t);
}

void RotCounterClockwise(cv::Mat &m)
{
  cv::transpose(m,m);
  cv::flip(m,m,0);
}
void RotClockwise(cv::Mat &m)
{
  cv::transpose(m,m);
  cv::flip(m,m,1);
}

// return 1:points are on a clockwise triangle, -1:counter-clockwise
// if the points are on a line, return -1 when p0 is center, 1 when p1 is center, 0 when p2 is center
int CheckClockwise(const Vector2f &p0, const Vector2f &p1, const Vector2f &p2)
{
  int dx1,dx2,dy1,dy2;
  dx1= p1.x-p0.x;
  dy1= p1.y-p0.y;
  dx2= p2.x-p0.x;
  dy2= p2.y-p0.y;

  if(dx1*dy2 > dy1*dx2 ) return 1;
  if(dx1*dy2 < dy1*dx2 ) return -1;
  if((dx1*dx2 <0) || (dy1*dy2 <0)) return -1;
  if((dx1*dx1 + dy1*dy1 < dx2*dx2 + dy2*dy2)) return 1;
  return 0;
}

cv::Mat LoadTemplate(const char *filename)
{
  cv::Mat img= cv::imread(filename,0);
  int tsize((img.cols<=img.rows) ? img.cols : img.rows);
  cv::resize (img, img, cv::Size(tsize,tsize), 0,0, CV_INTER_LINEAR);
  cv::threshold(img,img,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);
  return img;
}
void ARMarkerToClockwiseTetragon(const ARMarker &marker, TTetragon &t)
{
  if(CheckClockwise(marker.c1,marker.c2,marker.c3)==1)
  {
    t.c[0]= marker.c1;
    t.c[1]= marker.c2;
    t.c[2]= marker.c3;
    t.c[3]= marker.c4;
  }
  else
  {
    t.c[0]= marker.c1;
    t.c[1]= marker.c4;
    t.c[2]= marker.c3;
    t.c[3]= marker.c2;
  }
}
double CalcSimilarity(const TTetragon &marker, const cv::Mat &image, const cv::Mat &tmpl, int *direction=NULL)
{
  if(CheckClockwise(marker.c[0],marker.c[1],marker.c[2])!=1 || CheckClockwise(marker.c[0],marker.c[2],marker.c[3])!=1)
    return 0.0;
  cv::Point2f src[4];
  for(int i(0);i<4;++i)
  {
    if(std::isinf(marker.c[i].x) || std::isinf(marker.c[i].y))
      return 0.0;
    src[i]= cv::Point2f(marker.c[i].x,marker.c[i].y);
  }
  cv::Point2f dst[4];
  dst[0]= cv::Point2f(0,0);
  dst[1]= cv::Point2f(tmpl.cols,0);
  dst[2]= cv::Point2f(tmpl.cols,tmpl.rows);
  dst[3]= cv::Point2f(0,tmpl.rows);

  cv::Mat trans= cv::getPerspectiveTransform(src, dst);
  cv::Mat detected;
  cv::warpPerspective(image, detected, trans, cv::Size(tmpl.cols,tmpl.rows));

  cv::Mat tmp;
  cv::cvtColor(detected,tmp,CV_BGR2GRAY);
  detected= tmp;
  cv::threshold(detected,tmp,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);
  detected= tmp;

  cv::Mat matching;
  double similarity(0.0), s;
  for(int i(0);i<4;++i)
  {
    bitwise_xor(detected,tmpl,matching);
    s= 1-static_cast<double>(sum(matching)[0])/static_cast<double>(matching.cols*matching.rows);
    if(s>similarity)
    {
      similarity= s;
      if(direction)  *direction= i;
    }
    if(i<3)
      RotCounterClockwise(detected);
  }
  return similarity;
}


//-------------------------------------------------------------------------------------------


namespace nconst
{
  const double Dt(0.01);
  const double InitV(0.01);
  const double InitW(0.01);
//   const double InitF1(0.01);
//   const double InitF2(1.0);

const double InitF1(0.8);
const double InitF2(InitF1);

  const double NoiseCmn(0.2);
  const double NoiseC(NoiseCmn);
  const double NoiseR(0.5*NoiseCmn);
  const double NoiseV(2.0*NoiseCmn);
  const double NoiseW(1.0*NoiseCmn);

//   const double NoiseCmn(0.2);
//   const double NoiseC(0.2*NoiseCmn);
//   const double NoiseR(0.1*NoiseCmn);
//   const double NoiseV(NoiseCmn);
//   const double NoiseW(0.5*NoiseCmn);

  const double NoiseL1(0.001*NoiseCmn);
  const double NoiseL2(0.001*NoiseCmn);
//   const double NoiseF(0.01*NoiseCmn);

const double NoiseF(0.0*NoiseCmn);

  bool L1EqL2(true);

  const double ScaleX(500.0);
  const double ScaleY(500.0);
  const double Width(640.0);
  const double Height(480.0);
  const double WeightSigma(10.0);
  const double Epsilon(1.0e-6);
};

struct TParticle
{
  cv::Vec<double,3> c;
  cv::Vec<double,9> r_;
  cv::Mat_<double>  R;  // wrapper of r_
  cv::Vec<double,3> v, w;
  double l1,l2,f;

  TParticle() :
      c(0.,0.,0.),
      r_(0.,0.,0., 0.,0.,0., 0.,0.,0.),
      R(3,3,r_.val),
      v(0.,0.,0.),
      w(0.,0.,0.),
      l1(0.),
      l2(0.),
      f(0.)
    {}

  TParticle(const TParticle &p) :
      c(p.c),
      r_(p.r_),
      R(3,3,r_.val),
      v(p.v),
      w(p.w),
      l1(p.l1),
      l2(p.l2),
      f(p.f)
    {}

  const TParticle& operator=(const TParticle &rhs)
    {
#define XEQ(x_var) x_var= rhs.x_var;
      XEQ(c)
      XEQ(r_)
      XEQ(v)
      XEQ(w)
      XEQ(l1)
      XEQ(l2)
      XEQ(f)
#undef XEQ
      return *this;
    }
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
  cv::Vec<double,2> p[4];
};

TParticle operator+(const TParticle &lhs, const TParticle &rhs)
{
  TParticle res(lhs);
#define XEQ(x_var) res.x_var+= rhs.x_var;
  XEQ(c)
  XEQ(r_)
  XEQ(v)
  XEQ(w)
  XEQ(l1)
  XEQ(l2)
  XEQ(f)
#undef XEQ
  return res;
}

TParticle operator*(const TParticle &lhs, const double &rhs)
{
  TParticle res(lhs);
#define XEQ(x_var) res.x_var*= rhs;
  XEQ(c)
  XEQ(r_)
  XEQ(v)
  XEQ(w)
  XEQ(l1)
  XEQ(l2)
  XEQ(f)
#undef XEQ
  return res;
}

std::ostream& operator<<(std::ostream &lhs, const TParticle &rhs)
{
  lhs<<"c:"<<cv::Mat(rhs.c)<<" R v:"<<cv::Mat(rhs.v)<<" w:"<<cv::Mat(rhs.w)<<" l1:"<<rhs.l1<<" l2:"<<rhs.l2<<" f:"<<rhs.f;
  return lhs;
}

std::ostream& PrintParticle(std::ostream &lhs, const TParticle &rhs)
{
  int t(1);
  for(int i(0);i<3;++i,++t)  lhs<<" "<<rhs.c[i];
  lhs<<" #"<<t; ++t;
  for(int i(0);i<9;++i,++t)  lhs<<" "<<rhs.r_[i];
  lhs<<" #"<<t; ++t;
  for(int i(0);i<3;++i,++t)  lhs<<" "<<rhs.v[i];
  lhs<<" #"<<t; ++t;
  for(int i(0);i<3;++i,++t)  lhs<<" "<<rhs.w[i];
  lhs<<" #"<<t; ++t;
  lhs<<" "<<rhs.l1;
  lhs<<" "<<rhs.l2;
  lhs<<" "<<rhs.f;
  return lhs;
}

std::ostream& PrintParticle(std::ostream &lhs, const TParticleW &rhs)
{
  PrintParticle(lhs,rhs.P) << " #" << rhs.W;
}

std::ostream& operator<<(std::ostream &lhs, const TParticleW &rhs)
{
  lhs<<rhs.P<<"; W:"<<rhs.W;
  return lhs;
}

std::ostream& operator<<(std::ostream &lhs, const TObservation &rhs)
{
  for(int i(0);i<4;++i)  lhs<<" "<<cv::Mat(rhs.p[i]);
  return lhs;
}

template <typename t_elem>
inline cv::Mat_<t_elem> GetWedge (const  cv::Vec<t_elem,3> &w)
{
  cv::Mat_<t_elem> wedge(3,3);
  wedge(0,0)=0.0;    wedge(0,1)=-w(2);  wedge(0,2)=w(1);
  wedge(1,0)=w(2);   wedge(1,1)=0.0;    wedge(1,2)=-w(0);
  wedge(2,0)=-w(1);  wedge(2,1)=w(0);   wedge(2,2)=0.0;
  return wedge;
}

template <typename t_elem>
inline cv::Mat_<t_elem> Rodrigues (const cv::Vec<t_elem,3> &w)
{
  double th= norm(w);
  if(th<nconst::Epsilon)  return cv::Mat_<t_elem>::eye(3,3);
  cv::Mat_<t_elem> w_wedge(3,3);
  w_wedge= GetWedge(w *(1.0/th));
  return cv::Mat_<t_elem>::eye(3,3) + w_wedge * std::sin(th) + w_wedge * w_wedge * (1.0-std::cos(th));
}

template <typename t_elem>
inline cv::Vec<t_elem,3> InvRodrigues (const cv::Mat_<t_elem> &R)
{
  double alpha= (R(0,0)+R(1,1)+R(2,2) - 1.0) / 2.0;;

  if((alpha-1.0 < nconst::Epsilon) && (alpha-1.0 > -nconst::Epsilon))
    return cv::Vec<t_elem,3>(0.0,0.0,0.0);
  else
  {
    cv::Vec<t_elem,3> w;
    double th = std::acos(alpha);
    double tmp= 0.5 * th / std::sin(th);
    w[0] = tmp * (R(2,1) - R(1,2));
    w[1] = tmp * (R(0,2) - R(2,0));
    w[2] = tmp * (R(1,0) - R(0,1));
    return w;
  }
}

template <typename t_elem>
inline cv::Mat_<t_elem> AverageRotations (const cv::Mat_<t_elem> &R1, const cv::Mat_<t_elem> &R2, const t_elem &w2)
{
  cv::Vec<t_elem,3> w= InvRodrigues(cv::Mat_<double>(R2*R1.t()));
  return Rodrigues(w2*w)*R1;
}

void GenerateParticle(TParticle &p)
{
  p.c= cv::Vec<double,3>(Rand(-1.0,1.0), Rand(-1.0,1.0), Rand(0.0,1.0));
  cv::Vec<double,3> axis(Rand(-1.0,1.0), Rand(-1.0,1.0), Rand(-1.0,1.0));
  cv::normalize(axis,axis);
  Rodrigues(axis*static_cast<double>(Rand(-M_PI/2.0,M_PI/2.0))).copyTo(p.R);
  p.v= cv::Vec<double,3>(Rand(-nconst::InitV,nconst::InitV), Rand(-nconst::InitV,nconst::InitV), Rand(-nconst::InitV,nconst::InitV));
  p.w= cv::Vec<double,3>(Rand(-nconst::InitW,nconst::InitW), Rand(-nconst::InitW,nconst::InitW), Rand(-nconst::InitW,nconst::InitW));
  p.l1= Rand(0.01,1.0);
  p.l2= Rand(0.01,1.0);
  p.f= Rand(nconst::InitF1,nconst::InitF2);
}

void GenerateParticles(std::vector<TParticleW> &particles, int N)
{
  particles.resize(N);
  double w(1.0/static_cast<double>(N));
  for(std::vector<TParticleW>::iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
  {
    GenerateParticle(itr->P);
    itr->W= w;
  }
}

int BestParticleIdx(const std::vector<TParticleW> &particles)
{
  int i(0),id(-1);
  double maxw(-1.0);
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr,++i)
    if(itr->W > maxw)
    {
      maxw= itr->W;
      id= i;
    }
  return id;
}
// particles[id1].W(best) > particles[id2].W > others
void BestParticleIdx2(const std::vector<TParticleW> &particles, int &id1, int &id2)
{
  int i(0);
  id1=-1; id2=-1;
  double maxw1(-1.0),maxw2(-1.0);
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr,++i)
    if(itr->W>maxw2)
    {
      if(itr->W>maxw1)
      {
        maxw2= maxw1; id2= id1;
        maxw1= itr->W; id1= i;
      }
      else
      {
        maxw2= itr->W; id2= i;
      }
    }
}

TParticle AverageParticles(const std::vector<TParticleW> &particles)
{
  TParticle p;
  cv::Mat_<double> R(cv::Mat_<double>::eye(3,3)), eye(R);
  // cv::Vec<double,3> w(0.0,0.0,0.0);
  double weight(1.0);
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
  {
    p= p + itr->P * itr->W;
    // cv::Vec<double,3> w= InvRodrigues(itr->P.R);
    // R= R*Rodrigues(itr->W*w);
    // w+= itr->W*InvRodrigues(itr->P.R);
    R= AverageRotations(R, itr->P.R, weight);
    weight-= itr->W;
  }
  // Rodrigues(w).copyTo(p.R);
  R.copyTo(p.R);
  return p;
}

TParticle EstimateFromParticles(const std::vector<TParticleW> &particles)
{
  TParticle p;
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    p= p + itr->P * itr->W;
  int id1,id2;
  BestParticleIdx2(particles,id1,id2);
  AverageRotations(particles[id1].P.R, particles[id2].P.R, 0.5).copyTo(p.R);
  return p;
}

const TParticle& BestParticle(const std::vector<TParticleW> &particles)
{
  return particles[BestParticleIdx(particles)].P;
}

void PrintParticles(std::ostream &os, const std::vector<TParticleW> &particles)
{
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
    os<<*itr<<std::endl;
}

double SumWeights(const std::vector<TParticleW> &particles)
{
  double sumw(0.0);
  for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
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
next= curr;

#define NV3  cv::Vec<double,3>(NDRand(), NDRand(), NDRand())
  double sw(add_noise?1.0:0.0);
  next.c= curr.c + curr.v*nconst::Dt + sw*nconst::NoiseC*NV3;

  cv::Mat(Rodrigues(curr.w*nconst::Dt + sw*nconst::NoiseR*NV3)*curr.R).copyTo(next.R);
// cv::Mat(Rodrigues(curr.w*nconst::Dt + 10.0*sw*nconst::NoiseR*NV3)*curr.R).copyTo(next.R);
// cerr<<cv::norm(next.R.col(0))<<","<<cv::norm(next.R.col(1))<<","<<cv::norm(next.R.col(2))<<endl;
// cv::Mat(cv::Mat_<double>::eye(3,3)).copyTo(next.R);

  next.v= curr.v + sw*nconst::NoiseV*NV3;
  next.w= curr.w + sw*nconst::NoiseW*NV3;
  next.l1= curr.l1 + sw*nconst::NoiseL1*NDRand();
  next.l2= curr.l2 + sw*nconst::NoiseL2*NDRand();
  next.f= curr.f + sw*nconst::NoiseF*NDRand();

  if(next.l1<0.0)  next.l1= nconst::Epsilon;
  if(next.l2<0.0)  next.l2= nconst::Epsilon;
  if(next.f<0.0)  next.f= nconst::Epsilon;

  if(next.c[2]<next.f)  next.c[2]= next.f+nconst::Epsilon;

  if(nconst::L1EqL2)  next.l2= next.l1;
#undef NV3
}

void EstimateObservation(const TParticle &p, TObservation &o)
{
  cv::Vec<double,3> lp[4], wp;
  lp[0]= cv::Vec<double,3>(-p.l1/2.0, -p.l2/2.0, 0.0);
  lp[1]= cv::Vec<double,3>(+p.l1/2.0, -p.l2/2.0, 0.0);
  lp[2]= cv::Vec<double,3>(+p.l1/2.0, +p.l2/2.0, 0.0);
  lp[3]= cv::Vec<double,3>(-p.l1/2.0, +p.l2/2.0, 0.0);
  for(int i(0);i<4;++i)
  {
    cv::Mat(cv::Mat(p.c) + p.R * cv::Mat(lp[i])).copyTo(wp);
// cerr<<"wp[2]:"<<wp[2]<<endl;
    if(wp[2]<nconst::Epsilon)
    {
      o.p[i][0]= 0.0;
      o.p[i][1]= 0.0;
    }
    else
    {
      o.p[i][0]= nconst::ScaleX * wp[0] * p.f / wp[2] + 0.5*nconst::Width;
      o.p[i][1]= nconst::ScaleY * wp[1] * p.f / wp[2] + 0.5*nconst::Height;
    }
  }
}

double ComputeWeight(const TParticle &p, const TObservation &o)
{
  TObservation est;
  EstimateObservation(p,est);
  double w(nconst::Epsilon);
// cerr<<"p.v[2]:"<<p.v[2]<<endl;
// cerr<<"p.c[2]:"<<p.c[2]<<endl;
// cerr<<"o:"<<o<<endl;
// cerr<<"e:"<<est<<endl;
  if(CheckClockwise(ToVector2f(est.p[0]),ToVector2f(est.p[1]),ToVector2f(est.p[2]))==1)
  {
    // for(int i(0);i<4;++i)  w+= Square((o.p[i][0]-est.p[i][0])/nconst::ScaleX)+Square((o.p[i][1]-est.p[i][1])/nconst::ScaleY);
    // w= real_exp(-0.5*nconst::WeightSigma*(w))+nconst::Epsilon;

    // bool correct(true);
    // for(int i(0);i<4 && correct;++i)
    // {
    //   double d= Square((o.p[i][0]-est.p[i][0])/nconst::ScaleX)+Square((o.p[i][1]-est.p[i][1])/nconst::ScaleY);
    //   double d2;
    //   for(int j(0);j<4 && correct;++j)
    //     if(i!=j)
    //     {
    //       d2= Square((o.p[j][0]-est.p[i][0])/nconst::ScaleX)+Square((o.p[j][1]-est.p[i][1])/nconst::ScaleY);
    //       if(d>d2) correct= false;
    //     }
    //   w+= d;
    // }
    // if(correct)  w= real_exp(-0.5*nconst::WeightSigma*(w))+nconst::Epsilon;
    // else         w= nconst::Epsilon;

    for(int i(0);i<4;++i)
    {
      double d= Square((o.p[i][0]-est.p[i][0])/nconst::ScaleX)+Square((o.p[i][1]-est.p[i][1])/nconst::ScaleY);
      double d2;
      bool correct(true);
      for(int j(0);j<4 && correct;++j)
        if(i!=j)
        {
          d2= Square((o.p[j][0]-est.p[i][0])/nconst::ScaleX)+Square((o.p[j][1]-est.p[i][1])/nconst::ScaleY);
          if(d>d2) correct= false;
        }
      if(correct)  w+= d;
      else         w+= 20.0*d;
    }
    w= real_exp(-0.5*nconst::WeightSigma*(w))+nconst::Epsilon;

  }
cerr<<"w:"<<w<<endl;
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


//-------------------------------------------------------------------------------------------


int main (int argc, char * const argv[])
{
  unsigned seed ((unsigned)time(NULL));
  XSrand(seed);
  bool writeVideo = false;

  cv::namedWindow(WINDOW_NAME,1);
  cv::namedWindow("marker",1);

  cv::VideoCapture  camera;
  cv::VideoWriter writer;
  bool isColor   = true;
  double fps     = 15.0;  // or 30
  int frameW  = 640; // 744 for firewire cameras
  int frameH  = 480; // 480 for firewire cameras

  std::vector<TParticleW>  particles;
  GenerateParticles(particles, 10000);
//   GenerateParticles(particles, 8000);

  camera.open(0);
  if(!camera.isOpened())
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  if( writeVideo )
  {
    writer.open("out.mpeg",
                   // CV_FOURCC('P','I','M','1'),
                   CV_FOURCC('M','J','P','G'),
                   fps,cv::Size(frameW,frameH),isColor);
  }

  // template image
  cv::Mat template_image= LoadTemplate("marker3.png");
  cv::imshow("marker", template_image*255);

  // marker detection
  Buffer *buffer = new Buffer();
  EdgelDetector *edgelDetector = new EdgelDetector();

  edgelDetector->debugDrawLineSegments( false );
  edgelDetector->debugDrawPartialMergedLineSegments( false );
  edgelDetector->debugDrawMergedLineSegments( false );
  edgelDetector->debugDrawExtendedLineSegments( true );
  edgelDetector->debugDrawSectors( false );
  edgelDetector->debugDrawSectorGrids( false );
  edgelDetector->debugDrawEdges( false );
  edgelDetector->debugDrawCorners( false );

  edgelDetector->debugDrawMarkers( true );

  // get an initial frame and duplicate it for later work
  cv::Mat current_frame;
  camera >> current_frame;
  draw_image.create(cv::Size(current_frame.cols, current_frame.rows), CV_8UC3);

  // as long as there are images ...
  while(true)
  {
    // draw faces
    // cv::flip (current_frame, draw_image, 1);

    camera >> current_frame;
    if(current_frame.cols*current_frame.rows==0)
      {cerr<<"capture failed"<<endl; continue;}
    cv::resize (current_frame, draw_image, cv::Size(), 1.0,1.0, CV_INTER_LINEAR);


    // Perform a Gaussian blur
    //cv::Smooth( draw_image, draw_image, CV_GAUSSIAN, 3, 3 );

    buffer->setBuffer((unsigned char *) draw_image.data, draw_image.cols, draw_image.rows);

    edgelDetector->setBuffer(buffer);
    std::vector<ARMarker> markers = edgelDetector->findMarkers();

    /* *-/{
      double s;
      int d;
      TTetragon t;
      for(std::vector<ARMarker>::const_iterator itr(markers.begin()),last(markers.end());itr!=last;++itr)
      {
        ARMarkerToClockwiseTetragon(*itr,t);
        s= CalcSimilarity(t, current_frame, template_image, &d);
        if(s > 0.8)
        {
          cout<<" "<<s<<","<<d<<"("<<t.c[0].x<<","<<t.c[0].y<<")";
          for(int i(0);i<4;++i)
            edgelDetector->drawLine(t.c[i].x, t.c[i].y, t.c[(i+1)%4].x, t.c[(i+1)%4].y, 0, (d==i?255:0), 255, (d==i?10:2));
        }
      }
      // usleep(100000);
      cout<<endl;
    }//*/


    /* */{
      TObservation o, est;
      double s, maxs(0.0);
      int d;
      TTetragon t;
      for(std::vector<ARMarker>::const_iterator itr(markers.begin()),last(markers.end());itr!=last;++itr)
      {
        ARMarkerToClockwiseTetragon(*itr,t);
        s= CalcSimilarity(t, current_frame, template_image, &d);
        if(s > 0.8 && s>maxs)
        {
          for(int i(0);i<4;++i)
          {
            o.p[i][0]= t.c[(i+d)%4].x;
            o.p[i][1]= t.c[(i+d)%4].y;
          }
          maxs= s;
        }
      }

      bool observable= (maxs>0.8);

      if(observable)
        UpdateParticles(particles, o);
      else
        UpdateParticles(particles);


      if(observable)
        cout<<"obsv: "<<o<<endl;

      int skipper(0);
      for(std::vector<TParticleW>::const_iterator itr(particles.begin()),last(particles.end());itr!=last;++itr)
      {
        if(skipper!=0)  {--skipper; continue;}
        skipper= particles.size()/100;

        TObservation e;
        EstimateObservation(itr->P, e);
        // cout<<"e: "<<e<<endl;
        for(int i(0);i<4;++i)
          edgelDetector->drawLine(e.p[i][0], e.p[i][1], e.p[(i+1)%4][0], e.p[(i+1)%4][1], 0, (0==i?200:255), 255, 0.5);
      }

      if(observable)
        for(int i(0);i<4;++i)
          edgelDetector->drawLine(o.p[i][0], o.p[i][1], o.p[(i+1)%4][0], o.p[(i+1)%4][1], (0==i?255:0), 255, 0, ((0==i||1==i)?10:2));


      // TParticle pest= AverageParticles(particles);
      TParticle pest= EstimateFromParticles(particles);
      // TParticle pest= BestParticle(particles);
      cout<<"pest: "<<pest<<endl;
      EstimateObservation(pest, est);

      cout<<"est: ("<<est.p[0][0]<<","<<est.p[0][1]<<")"<<endl;
      for(int i(0);i<4;++i)
        edgelDetector->drawLine(est.p[i][0], est.p[i][1], est.p[(i+1)%4][0], est.p[(i+1)%4][1], 0, (0==i?200:0), 255, ((0==i||1==i)?10:2));

      static ofstream ofs_pest("res/pest.dat");
      PrintParticle(ofs_pest,pest)<<endl;


      TParticle pbest= BestParticle(particles);
      cout<<"pbest: "<<pbest<<endl;
      TObservation best;
      EstimateObservation(pbest, best);

      cout<<"best: ("<<best.p[0][0]<<","<<best.p[0][1]<<")"<<endl;
      for(int i(0);i<4;++i)
        edgelDetector->drawLine(best.p[i][0], best.p[i][1], best.p[(i+1)%4][0], best.p[(i+1)%4][1], 255, 0, (0==i?200:0), ((0==i||1==i)?10:2));


// cerr<<"P[0]:"<<particles[0].P<<endl;
// TObservation e0;EstimateObservation(particles[0].P, e0);
// cerr<<"e0: ("<<e0.p[0][0]<<","<<e0.p[0][1]<<")"<<endl;

      // ofstream ofs_part("res/particles.dat");
      // PrintParticles(ofs_part, particles);
    }//*/


    debugDrawAll();

    // just show the image
    cv::imshow(WINDOW_NAME, draw_image);

    if( writeVideo )
    {
      writer.write(draw_image);
    }

    // wait a tenth of a second for keypress and window drawing
    int key = cv::waitKey (10);
    if (key == 'q' || key == 'Q')
      break;

    switch( key ) {
      case '4':  edgelDetector->debugDrawLineSegments( !edgelDetector->drawLineSegments );
        break;
      case '5':  edgelDetector->debugDrawPartialMergedLineSegments( !edgelDetector->drawPartialMergedLineSegments );
        break;
      case '6':  edgelDetector->debugDrawMergedLineSegments( !edgelDetector->drawMergedLineSegments );
        break;
      case '7':  edgelDetector->debugDrawExtendedLineSegments( !edgelDetector->drawExtendedLineSegments );
        break;
      case '9':  edgelDetector->debugDrawMarkers( !edgelDetector->drawMarkers );
        break;
      case '1':  edgelDetector->debugDrawSectors( !edgelDetector->drawSectors );
        break;
      case '2':  edgelDetector->debugDrawSectorGrids( !edgelDetector->drawSectorGrids );
        break;
      case '3':  edgelDetector->debugDrawEdges( !edgelDetector->drawEdges );
        break;
      case '8':  edgelDetector->debugDrawCorners( !edgelDetector->drawCorners );
        break;
      default:
        break;
    }
  }

  // be nice and return no error
  return 0;
}
