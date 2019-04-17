//-------------------------------------------------------------------------------------------
/*! \file    segment_obj_simple1b.cpp
    \brief   Segment objects on a plate of specific color (e.g. white).
             Segmentation is based on non-white color detection.
             Working well.
             Class version of segment_obj_simple1.cpp
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.17, 2017

g++ -g -Wall -O2 -o segment_obj_simple1b.out segment_obj_simple1b.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

Based on:
  segment_obj_simple1.cpp
  find_white_contours.cpp
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

struct TObjectDetectorParams
{
  // For white detector:
  int WhiteSMax;
  int WhiteVMin;
  int NErode1;
  int NDilate1;

  // For objects-on-white detector:
  int ThreshS;
  int ThreshV;
  int NErode2;
  int NDilate2;
  int RectLenMin;
  int RectLenMax;
  double ContourApproxEps;  // Contour approximation. If zero, no approximation.

  TObjectDetectorParams();
};
void WriteToYAML(const std::vector<TObjectDetectorParams> &blob_params, const std::string &file_name);
void ReadFromYAML(std::vector<TObjectDetectorParams> &blob_params, const std::string &file_name);
//-------------------------------------------------------------------------------------------

class TObjectDetector
{
public:
  void Init();
  void Step(const cv::Mat &frame);
  void Draw(cv::Mat &frame);

  TObjectDetectorParams& Params()  {return params_;}
  const TObjectDetectorParams& Params() const {return params_;}

  const cv::Mat& WhiteMask() const {return mask_white_biggest_;}
  const cv::Mat& ObjectMask() const {return mask_objects_;}
  const std::vector<std::vector<cv::Point> >& Contours() const {return contours_obj_;}

private:
  TObjectDetectorParams params_;

  cv::Mat mask_white_biggest_;
  cv::Mat mask_objects_;
  std::vector<std::vector<cv::Point> > contours_obj_;
};
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
// Implementation
//-------------------------------------------------------------------------------------------

/*Find contours of white areas.
  frame: Input image.
  frame_white: Detected white image.
  contours: Found contours.
  v_min, s_max: Thresholds of V-minimum and S-maximum of HSV.
  n_dilate, n_erode: dilate and erode filter parameters before detecting contours.
*/
void FindWhiteContours(
    const cv::Mat &frame,
    cv::Mat &frame_white,
    std::vector<std::vector<cv::Point> > &contours,
    int v_min=100, int s_max=20, int n_dilate=1, int n_erode=1)
{
  cv::Mat frame_hsv;

  // White detection
  cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
  cv::inRange(frame_hsv, cv::Scalar(0, 0, v_min), cv::Scalar(255, s_max, 255), frame_white);

  if(n_dilate>0)  cv::dilate(frame_white,frame_white,cv::Mat(),cv::Point(-1,-1), n_dilate);
  if(n_erode>0)   cv::erode(frame_white,frame_white,cv::Mat(),cv::Point(-1,-1), n_erode);

  // Contour detection
  cv::findContours(frame_white, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}
//-------------------------------------------------------------------------------------------

// Make a mask from biggest contour.
void MakeBiggestContourMask(const std::vector<std::vector<cv::Point> > &contours,
    cv::Mat &mask, bool convex=false, int fill_value=1)
{
  if(contours.size()==0)  return;
  double a(0.0),a_max(0.0), i_max(0);
  for(int i(0),i_end(contours.size()); i<i_end; ++i)
  {
    a= cv::contourArea(contours[i],false);
    if(a>a_max)  {a_max= a;  i_max= i;}
  }
  if(!convex)
    cv::drawContours(mask, contours, i_max, fill_value, /*thickness=*/-1);
  else
  {
    std::vector<std::vector<cv::Point> > hull(1);
    cv::convexHull(contours[i_max], hull[0], /*clockwise=*/true);
    cv::drawContours(mask, hull, 0, fill_value, /*thickness=*/-1);
  }
}
//-------------------------------------------------------------------------------------------

TObjectDetectorParams::TObjectDetectorParams()
{
  // For white detector:
  WhiteSMax= 20;
  WhiteVMin= 100;
  NErode1= 1;
  NDilate1= 1;

  // For objects-on-white detector:
  ThreshS= 30;
  ThreshV= 224;
  NErode2= 1;
  NDilate2= 2;
  RectLenMin= 40;
  RectLenMax= 400;
  ContourApproxEps= 3.0;  // Contour approximation. If zero, no approximation.
}
//-------------------------------------------------------------------------------------------

void WriteToYAML(const std::vector<TObjectDetectorParams> &params, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs<<"ObjectDetector"<<"[";
  for(std::vector<TObjectDetectorParams>::const_iterator itr(params.begin()),itr_end(params.end()); itr!=itr_end; ++itr)
  {
    fs<<"{";
    #define PROC_VAR(x)  fs<<#x<<itr->x;
    PROC_VAR(WhiteSMax  );
    PROC_VAR(WhiteVMin  );
    PROC_VAR(NErode1    );
    PROC_VAR(NDilate1   );
    PROC_VAR(ThreshS    );
    PROC_VAR(ThreshV    );
    PROC_VAR(NErode2    );
    PROC_VAR(NDilate2   );
    PROC_VAR(RectLenMin );
    PROC_VAR(RectLenMax );
    PROC_VAR(ContourApproxEps );
    fs<<"}";
    #undef PROC_VAR
  }
  fs<<"]";
  fs.release();
}
//-------------------------------------------------------------------------------------------

void ReadFromYAML(std::vector<TObjectDetectorParams> &params, const std::string &file_name)
{
  params.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["ObjectDetector"];
  for(cv::FileNodeIterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
  {
    TObjectDetectorParams cf;
    #define PROC_VAR(x)  if(!(*itr)[#x].empty())  (*itr)[#x]>>cf.x;
    PROC_VAR(WhiteSMax  );
    PROC_VAR(WhiteVMin  );
    PROC_VAR(NErode1    );
    PROC_VAR(NDilate1   );
    PROC_VAR(ThreshS    );
    PROC_VAR(ThreshV    );
    PROC_VAR(NErode2    );
    PROC_VAR(NDilate2   );
    PROC_VAR(RectLenMin );
    PROC_VAR(RectLenMax );
    PROC_VAR(ContourApproxEps );
    #undef PROC_VAR
    params.push_back(cf);
  }
  fs.release();
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
// class TObjectDetector
//-------------------------------------------------------------------------------------------

void TObjectDetector::Init()
{
}
//-------------------------------------------------------------------------------------------

void TObjectDetector::Step(const cv::Mat &frame)
{
  cv::Mat mask_white;
  std::vector<std::vector<cv::Point> > contours_w;
  FindWhiteContours(frame, mask_white, contours_w,
        /*v_min=*/params_.WhiteVMin, /*s_max=*/params_.WhiteSMax,
        /*n_dilate=*/params_.NDilate1, /*n_erode=*/params_.NErode1);

  // Make a mask of biggest contour:
  mask_white_biggest_.create(mask_white.size(), CV_8UC1);
  mask_white_biggest_.setTo(0);
  MakeBiggestContourMask(contours_w, mask_white_biggest_, /*convex=*/true);

  // Detect objects-on-white
  cv::Mat frame_white, frame_white_hsv;
  frame.copyTo(frame_white, mask_white_biggest_);

  // Non-white detection
  cv::cvtColor(frame_white, frame_white_hsv, cv::COLOR_BGR2HSV);
  cv::inRange(frame_white_hsv, cv::Scalar(0, params_.ThreshS, 0),
              cv::Scalar(255, 255, params_.ThreshV), mask_objects_);
  mask_objects_.setTo(0, 1-mask_white_biggest_);

  cv::dilate(mask_objects_,mask_objects_,cv::Mat(),cv::Point(-1,-1), params_.NDilate2);
  cv::erode(mask_objects_,mask_objects_,cv::Mat(),cv::Point(-1,-1), params_.NErode2);

  // Find object contours
  if(params_.ContourApproxEps>0.0)
  {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask_objects_, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    contours_obj_.clear();
    contours_obj_.resize(contours.size());
    std::vector<std::vector<cv::Point> >::iterator oitr(contours_obj_.begin());
    for(std::vector<std::vector<cv::Point> >::const_iterator
        citr(contours.begin()),citr_end(contours.end()); citr!=citr_end; ++citr,++oitr)
    {
      cv::approxPolyDP(*citr, *oitr, params_.ContourApproxEps, /*closed=*/true);
    }
  }
  else
  {
    contours_obj_.clear();
    cv::findContours(mask_objects_, contours_obj_, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  }
}
//-------------------------------------------------------------------------------------------

void TObjectDetector::Draw(cv::Mat &frame)
{
  cv::Mat &img_disp(frame);
  cv::Mat mask_objectss[3]= {128.0*mask_white_biggest_,128.0*mask_white_biggest_,128.0*mask_white_biggest_+128.0*mask_objects_}, mask_objectsc;
  cv::merge(mask_objectss,3,mask_objectsc);
  img_disp+= mask_objectsc;

  if(contours_obj_.size()>0)
  {
    for(int ic(0),ic_end(contours_obj_.size()); ic<ic_end; ++ic)
    {
      // double area= cv::contourArea(contours_obj_[ic],false);
      cv::Rect bound= cv::boundingRect(contours_obj_[ic]);
      int bound_len= std::max(bound.width, bound.height);
      if(bound_len<params_.RectLenMin || bound_len>params_.RectLenMax)  continue;
      cv::drawContours(img_disp, contours_obj_, ic, CV_RGB(255,0,255), /*thickness=*/2, /*linetype=*/8);

      cv::rectangle(img_disp, bound, cv::Scalar(0,0,255), 2);
    }
  }
}
//-------------------------------------------------------------------------------------------

}  // loco_rabbits
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("camera", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("detected", CV_WINDOW_AUTOSIZE);

  TObjectDetector detector;

  // For white detector:
  cv::createTrackbar("white_s_max", "detected", &detector.Params().WhiteSMax, 255, NULL);
  cv::createTrackbar("white_v_min", "detected", &detector.Params().WhiteVMin, 255, NULL);
  cv::createTrackbar("n_dilate1", "detected", &detector.Params().NDilate1, 10, NULL);
  cv::createTrackbar("n_erode1", "detected", &detector.Params().NErode1, 10, NULL);

  // For objects-on-white detector
  cv::createTrackbar("thresh_s", "detected", &detector.Params().ThreshS, 255, NULL);
  cv::createTrackbar("thresh_v", "detected", &detector.Params().ThreshV, 255, NULL);
  cv::createTrackbar("n_dilate2", "detected", &detector.Params().NDilate2, 10, NULL);
  cv::createTrackbar("n_erode2", "detected", &detector.Params().NErode2, 10, NULL);
  cv::createTrackbar("rect_len_min", "detected", &detector.Params().RectLenMin, 600, NULL);
  cv::createTrackbar("rect_len_max", "detected", &detector.Params().RectLenMax, 600, NULL);

  detector.Init();

  for(int i(0);;++i)
  {
    cv::Mat frame;
    cap >> frame;

    detector.Step(frame);

    cv::Mat img_disp;
    img_disp= 0.3*frame;
    detector.Draw(img_disp);

    cv::imshow("camera", frame);
    cv::imshow("detected", img_disp);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
