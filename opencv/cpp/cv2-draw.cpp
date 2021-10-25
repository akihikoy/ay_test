// g++ -g -Wall -O2 -o cv2-draw.out cv2-draw.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc.hpp>
#endif
#include <iostream>
#include <vector>
// #include <lora/rand.h>
// using namespace loco_rabbits;

int main(int, char**)
{
  cv::VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
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
    cap >> frame; // get a new frame from camera
    if(f%10!=0)  continue;

    #define RAND_PT cv::Point(rng.uniform(0,640),rng.uniform(0,480))
    for(int i(0); i<5; ++i)
    {
      cv::circle(frame, RAND_PT, rng.uniform(1,300), colors[0]/*cv::Scalar(255,0,255)*/, 2);

      cv::line(frame, RAND_PT, RAND_PT, colors[1]/*cv::Scalar(255,0,255)*/, 2);
    }

    std::vector<std::vector<cv::Point> >  points(1);
    for(int p(0); p<5; ++p)
      points[0].push_back(RAND_PT);
    // points[0].push_back(points[0][0]);
    cv::fillPoly(frame, points, colors[2]);
    cv::polylines(frame, points, /*isClosed=*/false, colors[3], 2);

    cv::imshow("camera", frame);
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
