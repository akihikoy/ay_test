//-------------------------------------------------------------------------------------------
/*! \file    lk_mov_det.cpp
    \brief   Moving object detection using LK optical flow.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.14, 2015

    g++ -I -Wall lk_mov_det.cpp -o lk_mov_det.out -lopencv_core -lopencv_ml -lopencv_video -lopencv_imgproc -lopencv_legacy -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

class TMovingObjectDetector
{
public:
  TMovingObjectDetector();
  void Step(const cv::Mat &frame);
  void Draw(cv::Mat &img);

private:
  cv::TermCriteria term_criteria_;
  cv::Size win_size_;
  int max_feat_count_;
  int reset_count_;

  cv::Mat gray_, prev_gray_;
  std::vector<cv::Point2f> points_[2];  // Feature points.
  std::vector<cv::Point2f> flow_;
  std::vector<uchar> status_;
  int count_;
};
//-------------------------------------------------------------------------------------------

TMovingObjectDetector::TMovingObjectDetector()
  :
    term_criteria_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03),
    win_size_(10,10),
    max_feat_count_(500),
    reset_count_(100),
    count_(0)
{
}
//-------------------------------------------------------------------------------------------

void TMovingObjectDetector::Step(const cv::Mat &frame)
{
  cv::cvtColor(frame, gray_, CV_BGR2GRAY);
  if(prev_gray_.empty())  gray_.copyTo(prev_gray_);

  if(points_[0].empty() || count_==0)
  {
    // Automatically detect feature points
    cv::goodFeaturesToTrack(prev_gray_, points_[0], max_feat_count_, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(prev_gray_, points_[0], win_size_, cv::Size(-1,-1), term_criteria_);
    count_= reset_count_;
  }
  if(!points_[0].empty())
  {
    // Do tracking
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_gray_, gray_, points_[0], points_[1], status_, err, win_size_, 3, term_criteria_, 0);
    // Keep only tracked points and compute flow vector
    size_t k(0);
    flow_.resize(points_[1].size());
    for(size_t i(0),i_end(points_[1].size()); i<i_end; ++i)
    {
      if(!status_[i])  continue;  // Corresponding feature not found
      points_[1][k]= points_[1][i];
      flow_[k]= points_[1][i] - points_[0][i];
      ++k;
    }
    points_[1].resize(k);
    flow_.resize(k);
  }
  std::swap(points_[1], points_[0]);
  cv::swap(prev_gray_, gray_);

  --count_;
}
//-------------------------------------------------------------------------------------------

void TMovingObjectDetector::Draw(cv::Mat &img)
{
  const std::vector<cv::Point2f> &points(points_[0]);
  if(points.empty() || flow_.empty())  return;
  for(size_t i(0),i_end(points.size()); i<i_end; ++i)
  {
    cv::circle(img, points[i], 3, cv::Scalar(0,0,255), 1, 8);
    float f= cv::norm(flow_[i]);
// std::cerr<<" "<<f;
    if(f<1.0)  continue;  // Not moving
    // if(f>50.0) continue;  // Outliers
    f*= 5.0;
    if(f>20.0)  f= 20.0;
    cv::circle(img, points[i], f, cv::Scalar(0,255,0), 1, 8);
    cv::line(img, points[i], points[i]+flow_[i], cv::Scalar(255,128,0), 2, 8);
  }
// std::cerr<<std::endl;
}
//-------------------------------------------------------------------------------------------


}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  TMovingObjectDetector mov_det;

  cv::namedWindow( "LKMovDet", 1 );

  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    mov_det.Step(frame);
    mov_det.Draw(frame);


    cv::imshow("LKMovDet", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }


  return 0;
}
//-------------------------------------------------------------------------------------------
