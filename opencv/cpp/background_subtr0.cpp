#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // medianBlur
// #include <opencv2/photo/photo.hpp>  // fastNlMeansDenoising
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include "cap_open.h"

// g++ -I -Wall background_subtr0.cpp -o background_subtr0.out -lopencv_core -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_photo -lopencv_highgui

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  // http://stackoverflow.com/questions/21873757/opencv-c-how-to-slow-down-background-adaptation-of-backgroundsubtractormog
  // cv::createBackgroundSubtractorMOG2 bkg_sbtr= cv::BackgroundSubtractorMOG(/*history=*/200, /*nmixtures=*/5, /*double backgroundRatio=*/0.7, /*double noiseSigma=*/0);
  cv::Ptr<cv::BackgroundSubtractorMOG2> bkg_sbtr= cv::createBackgroundSubtractorMOG2(/*history=*/30, /*varThreshold=*/10.0, /*detectShadows=*/true);

  const char *window("Background Subtraction");
  cv::namedWindow(window,1);
  int history(30);
  cv::createTrackbar( "History:", window, &history, 100, NULL);
  // cv::namedWindow("camera",1);
  // cv::namedWindow("mask",1);

  cv::Mat frame, mask, frame_masked;
  bool running(true);
  for(int i(0);;++i)
  {
    if(running)
    {
      cap >> frame; // get a new frame from camera

      // cv::medianBlur(frame, frame, 3);
      // cv::GaussianBlur(frame, frame, /*ksize=*/cv::Size(5,5), /*sigmaX=*/2.0);

      bkg_sbtr->apply(frame,mask,1./float(history));

      // cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      // cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);

      // frame_masked= cv::Scalar(0.0,0.0,0.0);
      // frame.copyTo(frame_masked, mask);
      frame_masked= 0.3*frame;
      cv::Mat masks[3]= {mask, 0.5*mask, 0.0*mask}, cmask;
      cv::merge(masks,3,cmask);
      frame_masked+= cmask;

      // cv::imshow(window, frame);
      cv::imshow(window, frame_masked);
      // cv::imshow("mask", mask);
    }

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  running= !running;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
