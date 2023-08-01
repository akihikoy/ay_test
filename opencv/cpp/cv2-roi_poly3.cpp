//-------------------------------------------------------------------------------------------
/*! \file    cv2-roi_poly3.cpp
    \brief   Polygon ROI test.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.01, 2023

g++ -g -Wall -O2 -o cv2-roi_poly3.out cv2-roi_poly3.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
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
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

  cv::RNG rng(cv::getTickCount());
  std::vector<std::vector<cv::Point> > points(1), hull(1);
  cv::namedWindow("camera",1);
  cv::namedWindow("img_roi",1);
  cv::Mat frame;
  for(int f(0);;++f)
  {
    cap >> frame; // get a new frame from camera

    if(f%10==0)
    {
      points[0].resize(6);
      for(int i(0);i<6;++i)
      {
        points[0][i]= cv::Point(rng.uniform(0,320), rng.uniform(0,240));
      }
    }

    cv::convexHull(points[0], hull[0]/*, bool clockwise=false, bool returnPoints=true */);

    cv::Rect bb= cv::boundingRect(hull[0]);
    std::vector<std::vector<cv::Point> > hull_shifted(1);
    hull_shifted[0].resize(hull[0].size());
    for(int i(0),i_end(hull[0].size());i<i_end;++i)  hull_shifted[0][i]= hull[0][i]-bb.tl();
    cv::Mat mask(bb.size(), CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, hull_shifted, cv::Scalar(255));
    cv::Mat img_roi;
    frame(bb).copyTo(img_roi, mask);
    //std::cerr<<"hull= "<<hull[0]<<std::endl;
    //std::cerr<<"hull_shifted= "<<hull_shifted[0]<<std::endl;

    // cv::fillPoly(frame, hull, cv::Scalar(255,255,0));
    cv::polylines(frame, hull, /*isClosed=*/true, cv::Scalar(255,255,0), 2);

    cv::imshow("camera", frame);
    cv::imshow("img_roi", img_roi);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
