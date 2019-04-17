//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_detector3.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.06, 2016

g++ -g -Wall -O2 -o simple_blob_detector3.out simple_blob_detector3.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cap_open.h"
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

bool ParamChanged(false);
void OnTrack(int,void*)
{
  ParamChanged= true;
}

int filterByColor = 0;
int minThreshold = 5;
int maxThreshold = 200;
int minArea = 40;
int minCircularity = 10;
int minConvexity = 87;
int minInertiaRatio = 1;
int threshold_value = 30;

void AssignParams(cv::SimpleBlobDetector::Params &params)
{
  #define R100(x)  ((double)x*0.01)
  params.filterByColor= filterByColor;
  params.blobColor= 0;

  // Change thresholds
  params.minThreshold = minThreshold;
  params.maxThreshold = maxThreshold;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = minArea;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = R100(minCircularity);

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = R100(minConvexity);

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = R100(minInertiaRatio);
}

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>=2)?(argv[1]):"0"), /*width=*/0, /*height=*/0))  return -1;

  // set resolution
  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);


  // Setup SimpleBlobDetector parameters.
  cv::SimpleBlobDetector::Params params;
  AssignParams(params);

  // Set up the detector with default parameters.
  cv::Ptr<cv::SimpleBlobDetector> detector;
  detector= new cv::SimpleBlobDetector(params);

  // Detect blobs.
  std::vector<cv::KeyPoint> keypoints;

  std::string win("camera");
  cv::namedWindow(win,1);

  cv::createTrackbar("filterByColor", win, &filterByColor, 1, OnTrack);
  cv::createTrackbar("minThreshold", win, &minThreshold, 255, OnTrack);
  cv::createTrackbar("maxThreshold", win, &maxThreshold, 255, OnTrack);
  cv::createTrackbar("minArea", win, &minArea, 5000, OnTrack);
  cv::createTrackbar("minCircularity", win, &minCircularity, 100, OnTrack);
  cv::createTrackbar("minConvexity", win, &minConvexity, 100, OnTrack);
  cv::createTrackbar("minInertiaRatio", win, &minInertiaRatio, 100, OnTrack);
  cv::createTrackbar("threshold_value", win, &threshold_value, 255, OnTrack);

  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    if(ParamChanged)
    {
      AssignParams(params);
      detector= new cv::SimpleBlobDetector(params);
      ParamChanged= false;
    }

    cv::cvtColor(frame, frame, CV_BGR2GRAY);
    cv::threshold(frame, frame, threshold_value, 255, cv::THRESH_BINARY_INV);
    // cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), 1);
    // cv::erode(frame,frame,cv::Mat(),cv::Point(-1,-1), 2);
    // cv::dilate(frame,frame,cv::Mat(),cv::Point(-1,-1), 1);

    detector->detect(frame, keypoints);
    cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
