/*! \file    optflow.cpp
    \brief   オプティカルフロー(実装) */
//------------------------------------------------------------------------------
#include "optflow.h"
#include <opencv2/legacy/legacy.hpp>
// NOTE: necessary for cvCalcOpticalFlowBM as it is legacy (there is a new implementation)
//------------------------------------------------------------------------------
using namespace std;

//==============================================================================
// class TOpticalFlow
//==============================================================================

TOpticalFlow::TOpticalFlow()
  :
    first_exec_   (true),
    block_size_   (cv::Size(8,8)),
    shift_size_   (cv::Size(20,20)),
    max_range_    (cv::Size(25,25)),
    use_previous_ (false),
    rows_         (0),
    cols_         (0)
{
}
//------------------------------------------------------------------------------

void TOpticalFlow::CalcBM(const cv::Mat &prev, const cv::Mat &curr)
{
  rows_= floor(static_cast<double>(curr.rows-block_size_.height)
                / static_cast<double>(shift_size_.height));
  cols_= floor(static_cast<double>(curr.cols-block_size_.width)
                / static_cast<double>(shift_size_.width));

  velx_.create(rows_, cols_, CV_32FC1);
  vely_.create(rows_, cols_, CV_32FC1);

  int use_prev= ((!first_exec_ && use_previous_)? 1 : 0);
  if (use_prev)
  {
    velx_*=0.5;
    vely_*=0.5;
  }

  CvMat prev2(prev), curr2(curr), velx2(velx_), vely2(vely_);
  cvCalcOpticalFlowBM (&prev2, &curr2, block_size_, shift_size_, max_range_,
    use_prev, &velx2, &vely2);
  first_exec_= false;
}
//------------------------------------------------------------------------------

void TOpticalFlow::DrawOnImg(cv::Mat &img, const cv::Scalar &color,
  int thickness, int line_type, int shift)
{
  int dx,dy;
  for (int i(0); i<cols_; i+=1)
  {
    for (int j(0); j<rows_; j+=1)
    {
      dx = cvRound(VelXAt(i,j));
      dy = cvRound(VelYAt(i,j));
      cv::line (img, GetPosOnImg(i,j), GetPosOnImg(i,j)+cv::Point(dx,dy),
              color, thickness, line_type, shift);
    }
  }
}
