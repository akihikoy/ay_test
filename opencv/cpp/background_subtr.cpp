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

// g++ -I -Wall background_subtr.cpp -o background_subtr.out -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_photo -lopencv_highgui

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  // http://stackoverflow.com/questions/21873757/opencv-c-how-to-slow-down-background-adaptation-of-backgroundsubtractormog
  // cv::createBackgroundSubtractorMOG2 bkg_sbtr= cv::BackgroundSubtractorMOG(/*history=*/200, /*nmixtures=*/5, /*double backgroundRatio=*/0.7, /*double noiseSigma=*/0);
  cv::Ptr<cv::BackgroundSubtractorMOG2> bkg_sbtr= cv::createBackgroundSubtractorMOG2(/*int history=*/10, /*double varThreshold=*/5.0, /*bool detectShadows=*/true);

  cv::namedWindow("camera",1);
  cv::namedWindow("mask",1);
  cv::Mat frame, mask, mask2, frame_masked;
  bool running(true);
  for(int i(0);;++i)
  {
    if(running)
    {
      cap >> frame; // get a new frame from camera

      // cv::medianBlur(frame, frame, 3);
      // cv::GaussianBlur(frame, frame, /*ksize=*/cv::Size(5,5), /*sigmaX=*/2.0);


      bkg_sbtr->apply(frame,mask);

      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 1);
      // cv::fastNlMeansDenoising(mask,mask,/*h=*/3.0 /*, templateWindowSize=7, searchWindowSize=21*/);  #Very slow

      // {std::stringstream file_name;
      // file_name<<"frame/frame"<<std::setfill('0')<<std::setw(4)<<i<<".jpg";
      // cv::imwrite(file_name.str(), frame);
      // std::cout<<"Saved "<<file_name.str()<<std::endl;}
      // {std::stringstream file_name;
      // file_name<<"frame/mask"<<std::setfill('0')<<std::setw(4)<<i<<".jpg";
      // cv::imwrite(file_name.str(), mask);
      // std::cout<<"Saved "<<file_name.str()<<std::endl;}
      // if(i==10000)  i=0;

      frame_masked= cv::Scalar(0.0,0.0,0.0);
      frame.copyTo(frame_masked, mask);
      // cv::imshow("camera", frame);
      cv::imshow("camera", frame_masked);

      // Draw contours:
      // cf. http://docs.opencv.org/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
      std::vector<std::vector<cv::Point> > contours;
      mask2= mask.clone();
      cv::findContours(mask2,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
      cv::cvtColor(mask,mask,CV_GRAY2RGB);
      for( int i = 0; i< contours.size(); i++ )
      {
        double area= cv::contourArea(contours[i]);
        std::cerr<<"area= "<<area<<std::endl;
        // Remove small and big area:
        if(area<10 || 0.05*(double(mask.rows*mask.cols))<area)
        {
          const cv::Point *pts= (const cv::Point*) cv::Mat(contours[i]).data;
          int npts= cv::Mat(contours[i]).rows;
          cv::fillPoly(mask, &pts, &npts, /*ncontours=*/1, CV_RGB(0,0,128), /*lineType=*/8);
        }
        else
          cv::drawContours(mask, contours, i, CV_RGB(255,0,0), /*thickness=*/1, /*linetype=*/8);
      }

      cv::imshow("mask", mask);
    }

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  running= !running;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
