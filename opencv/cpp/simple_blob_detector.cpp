//-------------------------------------------------------------------------------------------
/*! \file    simple_blob_detector.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.06, 2016

g++ -g -Wall -O2 -o simple_blob_detector.out simple_blob_detector.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
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

int main(int argc, char**argv)
{
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // set resolution
  // cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  // cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);


  // Setup SimpleBlobDetector parameters.
  cv::SimpleBlobDetector::Params params;
  // params.filterByColor= true;
  // params.blobColor= 0;

  // Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 200;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 10;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.1;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;

  // Set up the detector with default parameters.
  cv::Ptr<cv::SimpleBlobDetector> detector= cv::SimpleBlobDetector::create(params);

  // Detect blobs.
  std::vector<cv::KeyPoint> keypoints;

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    detector->detect(frame, keypoints);
    cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
