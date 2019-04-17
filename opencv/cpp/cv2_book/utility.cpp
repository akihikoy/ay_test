/*! \file    utility.cpp
    \brief   ユーティリティ関数群(実装) */
//------------------------------------------------------------------------------
#include <cmath>
#include "utility.h"
//------------------------------------------------------------------------------
#define _S cv::Scalar
static const cv::Scalar NUMBERED_COLORS[]
    = {_S(0,0,255),_S(0,255,0),_S(255,0,0),_S(255,255,0),_S(255,0,255),
      _S(255,255,255),_S(0,128,128),_S(128,128,0),_S(128,0,128),_S(0,0,128),
      _S(0,128,0),_S(128,0,0)};
#undef _S

const cv::Scalar& NumberedColor(unsigned int i)
{
  const int N= sizeof(NUMBERED_COLORS)/sizeof(NUMBERED_COLORS[0]);
  return NUMBERED_COLORS[i % N];
}
//------------------------------------------------------------------------------

void DrawGaussianEllipse(cv::Mat &img, const cv::Mat &cov,
    const double &meanx, const double &meany,
    const double &xscale, const double &yscale, const cv::Point &offset,
    const cv::Scalar &color, int thickness, int line_type, int shift, int N)
{
  const double a(cov.at<double>(0,0)), b(cov.at<double>(0,1)),
               c(cov.at<double>(1,1));
  const double p11(std::sqrt(a)), p21(b/std::sqrt(a)), p22(std::sqrt(c-b*b/a));
  const double step(2.0*M_PI/static_cast<double>(N));
  bool first(true);
  cv::Point p1,p2;
  for (double t(0.0); t<=2.000001*M_PI; t+=step)
  {
    p2=p1;
    p1.x= xscale*(meanx + p11*std::cos(t));
    p1.y= yscale*(meany + p21*std::cos(t)+p22*std::sin(t));
    p1=p1+offset;
    if(first)  {first=false; continue;}
    cv::line (img, p2, p1, color, thickness, line_type, shift);
  }
}
//------------------------------------------------------------------------------

void DrawRegularPolygon(int N, cv::Mat &img, const cv::Point &pos, int radius,
    const cv::Scalar &color, int thickness, int line_type, int shift)
{
  bool first(true);
  cv::Point p1,p2;
  for(double theta(0.0),d(2.0*M_PI/static_cast<double>(N));
      theta<2.000001*M_PI; theta+=d)
  {
    p2=p1;
    p1.x= radius*std::sin(theta);
    p1.y= radius*std::cos(theta);
    p1=p1+pos;
    if(first)  {first=false; continue;}
    cv::line (img, p2, p1, color, thickness, line_type, shift);
  }
}
