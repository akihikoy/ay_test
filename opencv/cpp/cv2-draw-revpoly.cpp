//-------------------------------------------------------------------------------------------
/*! \file    cv2-draw-revpoly.cpp
    \brief   Draw reversed filled polygon.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.12, 2022

g++ -g -Wall -O2 -o cv2-draw-revpoly.out cv2-draw-revpoly.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }

  cv::RNG rng(0xFFFFFFFF);

  #define _S cv::Scalar
  const cv::Scalar colors[] = {_S(0,0,255),_S(0,255,0),_S(255,0,0),_S(0,255,255),_S(255,255,0),_S(255,0,255),_S(255,255,255),
                              _S(0,128,128),_S(128,128,0),_S(128,0,128),_S(0,0,128),_S(0,128,0),_S(128,0,0)};
  #undef _S

  cv::namedWindow("camera",1);
  cv::Mat frame;
  for(int f(0);;++f)
  {
    cap >> frame;
    if(f%10!=0)  continue;

    #define RAND_PT cv::Point(rng.uniform(0,640),rng.uniform(0,480))

    std::vector<std::vector<cv::Point> >  points(1);
    for(int p(0); p<5; ++p)
      points[0].push_back(RAND_PT);
    cv::Mat revpoly(frame.size(), CV_8UC1);
    revpoly.setTo(1);
    cv::fillPoly(revpoly, points, 0);
    frame.setTo(colors[2], revpoly);

    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
