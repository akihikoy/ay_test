// ref. http://abhishek4273.com/tag/calcmotiongradient/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <cstdio>
#include <ctime>
#include "cap_open.h"

// g++ -I -Wall motion_history.cpp -o motion_history.out -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_imgproc -lopencv_legacy -lopencv_highgui -lopencv_videoio


void draw_motion_comp(cv::Mat& img, int x_coordinate, int y_coordinate, int width, int height, double angle, cv::Mat& result)
{
  // cv::rectangle(img,cv::Point(x_coordinate,y_coordinate), cv::Point(x_coordinate+width,y_coordinate+width), cv::Scalar(255,0,0), 1, 8, 0);
  int r,cx,cy;
  if(height/2 <= width/2)
    r = height/2;
  else
    r = width/2;
  cx = x_coordinate + width/2;
  cy = y_coordinate + height/2;
  angle = angle*M_PI/180;
  cv::circle(img, cv::Point(cx,cy), r, cv::Scalar(255,0,0),1, 8, 0);
  cv::line(img, cv::Point(cx,cy), cv::Point(int(cx+std::cos(angle)*r), int(cy+std::sin(angle)*r)), cv::Scalar(255,0,0), 1, 8, 0);
  result = img.clone();
}

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  float MHI_DURATION = 0.05;
  int DEFAULT_THRESHOLD = 32;
  float MAX_TIME_DELTA = 12500.0;
  float MIN_TIME_DELTA = 5;

  cv::namedWindow("camera",1);

  cv::Mat frame,ret,frame_diff,gray_diff,motion_mask;
  cap >> frame;
  int h(frame.size().height), w(frame.size().width);
  cv::Mat prev_frame(frame.clone());
  cv::Mat motion_history(h,w, CV_32FC1,cv::Scalar(0,0,0));
  cv::Mat hsv(h,w, CV_8UC3,cv::Scalar(0,255,0));
  cv::Mat mg_mask(h,w, CV_8UC1,cv::Scalar(0,0,0));
  cv::Mat mg_orient(h,w, CV_32FC1,cv::Scalar(0,0,0));
  cv::Mat seg_mask(h,w, CV_32FC1,cv::Scalar(0,0,0));
  std::vector<cv::Rect> seg_bounds;
  cv::Mat silh_roi,orient_roi,mask_roi,mhi_roi;

  // cv::Ptr<cv::BackgroundSubtractorMOG2> bkg_sbtr= cv::createBackgroundSubtractorMOG2(/*int history=*/10, /*double varThreshold=*/5.0, /*bool detectShadows=*/true);

  for(;;)
  {
    cap >> frame;

    ret = frame.clone();
    absdiff(frame, prev_frame, frame_diff);
    cvtColor(frame_diff,gray_diff, CV_BGR2GRAY );
    threshold(gray_diff,ret,DEFAULT_THRESHOLD,255,0);
    motion_mask = ret.clone();
    // bkg_sbtr->apply(frame,motion_mask);

    double timestamp= 1000.0*clock()/CLOCKS_PER_SEC;
    cv::updateMotionHistory(motion_mask,motion_history,timestamp,MHI_DURATION);
    cv::calcMotionGradient(motion_history, mg_mask, mg_orient, 5, 12500.0, 3);
    cv::segmentMotion(motion_history, seg_mask, seg_bounds, timestamp, 32);

    prev_frame = frame.clone();

    for(unsigned int h = 0; h < seg_bounds.size(); h++)
    {
      cv::Rect rec = seg_bounds[h];
      if(rec.area() > 5000 && rec.area() < 70000)
      {
        silh_roi = motion_mask(rec);
        orient_roi = mg_orient(rec);
        mask_roi = mg_mask(rec);
        mhi_roi = motion_history(rec);
        if(cv::norm(silh_roi, cv::NORM_L2, cv::noArray()) > rec.area()*0.5)
        {
          double angle = cv::calcGlobalOrientation(orient_roi, mask_roi, mhi_roi,timestamp, MHI_DURATION);
          // if((60.<angle&&angle<120.) || (240.<angle&&angle<300.))
          {
            std::cout << angle << std::endl;
            cv::rectangle(frame, rec,cv::Scalar(0,0,255),3);
            draw_motion_comp(frame, rec.x, rec.y, rec.width, rec.height,angle,frame);
          }
        }
      }
    }

    cv::imshow("camera", frame);
    char c(cv::waitKey(30));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
