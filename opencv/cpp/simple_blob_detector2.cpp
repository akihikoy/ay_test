//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_detector2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.06, 2016

g++ -g -Wall -O2 -o simple_blob_detector2.out simple_blob_detector2.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
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

double ZoomLevels[]={1.0,2.0,3.0};
int ZoomIdx(0);

bool ParamChanged(false);
void OnTrack(int,void*)
{
  ParamChanged= true;
}

int filterByColor = 0;
int minThreshold = 5;
int maxThreshold = 200;
int minArea = 40;
int maxArea = 150;
int minCircularity = 10;
int minConvexity = 87;
int minInertiaRatio = 1;

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
  params.minArea = (minArea>0 ? minArea : 1);
  params.maxArea = (maxArea>0 ? maxArea : 1);

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
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  // set resolution
  // cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);


  // Setup SimpleBlobDetector parameters.
  cv::SimpleBlobDetector::Params params;
  AssignParams(params);

  // Set up the detector with default parameters.
  cv::Ptr<cv::SimpleBlobDetector> detector;
  detector= cv::SimpleBlobDetector::create(params);

  // Detect blobs.
  std::vector<cv::KeyPoint> keypoints;

  std::string win("camera");
  cv::namedWindow(win,1);

  cv::createTrackbar("filterByColor", win, &filterByColor, 1, OnTrack);
  cv::createTrackbar("minThreshold", win, &minThreshold, 255, OnTrack);
  cv::createTrackbar("maxThreshold", win, &maxThreshold, 255, OnTrack);
  cv::createTrackbar("minArea", win, &minArea, 5000, OnTrack);
  cv::createTrackbar("maxArea", win, &maxArea, 5000, OnTrack);
  cv::createTrackbar("minCircularity", win, &minCircularity, 100, OnTrack);
  cv::createTrackbar("minConvexity", win, &minConvexity, 100, OnTrack);
  cv::createTrackbar("minInertiaRatio", win, &minInertiaRatio, 100, OnTrack);

  cv::Mat frame;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    if(ParamChanged)
    {
      AssignParams(params);
      detector= cv::SimpleBlobDetector::create(params);
      ParamChanged= false;
    }

    detector->detect(frame, keypoints);

    // Zoom:
    if(ZoomLevels[ZoomIdx]!=1.0)
    {
      cv::resize(frame,frame,cv::Size(frame.cols*ZoomLevels[ZoomIdx],frame.rows*ZoomLevels[ZoomIdx]));
      for(std::vector<cv::KeyPoint>::iterator itr(keypoints.begin()),itr_end(keypoints.end()); itr!=itr_end; ++itr)
      {
        itr->pt.x*= ZoomLevels[ZoomIdx];
        itr->pt.y*= ZoomLevels[ZoomIdx];
        itr->size*= ZoomLevels[ZoomIdx];
      }
    }
    // Emphasize size visualization:
    for(std::vector<cv::KeyPoint>::iterator itr(keypoints.begin()),itr_end(keypoints.end()); itr!=itr_end; ++itr)
      itr->size*= 3.0;

    cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c=='z')
    {
      ZoomIdx++;
      if(ZoomIdx>=int(sizeof(ZoomLevels)/sizeof(ZoomLevels[0])))  ZoomIdx=0;
    }
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
