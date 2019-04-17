//-------------------------------------------------------------------------------------------
/*! \file    stereo_flow2.cpp
    \brief   Matching calculation for rectified stereo images.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.07, 2016

g++ -g -Wall -O2 -o stereo_flow2.out stereo_flow2.cpp -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui
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

template<typename t_stereo>
cv::Mat FlowStereo(int we, int wd, cv::Mat frame1, cv::Mat frame2, t_stereo &stereo)
{
  // frame1*= 200;
  // frame2*= 200;
  cv::erode(frame1,frame1,cv::Mat(),cv::Point(-1,-1), we);
  cv::dilate(frame1,frame1,cv::Mat(),cv::Point(-1,-1), wd);
  cv::erode(frame2,frame2,cv::Mat(),cv::Point(-1,-1), we);
  cv::dilate(frame2,frame2,cv::Mat(),cv::Point(-1,-1), wd);

  cv::Mat disparity;
  stereo(frame1, frame2, disparity, /*disptype=*/CV_32FC1);
  return disparity;
}
//-------------------------------------------------------------------------------------------

void FlowStereo2(int we, int wd, cv::Mat &frame1, cv::Mat &frame2)
{
  int y_filter(32), y_step(16), th_match(16);

  // Remove noise, make remaining pixels bigger:
  cv::erode(frame1,frame1,cv::Mat(),cv::Point(-1,-1), we);
  cv::dilate(frame1,frame1,cv::Mat(),cv::Point(-1,-1), wd);
  cv::erode(frame2,frame2,cv::Mat(),cv::Point(-1,-1), we);
  cv::dilate(frame2,frame2,cv::Mat(),cv::Point(-1,-1), wd);
  // Vertical filter to make detecting flow easier:
  cv::Mat kernel(cv::Size(1,y_filter),CV_32F);
  kernel= cv::Mat::ones(kernel.size(),CV_32F)/(float)(kernel.rows*kernel.cols);
    // TODO:FIXME: kernel can be stored (does not change)
  cv::filter2D(frame1, frame1, /*ddepth=*/-1, kernel);
  cv::filter2D(frame2, frame2, /*ddepth=*/-1, kernel);

  // frame1*= 200;
  // frame2*= 200;
  cv::Mat seg1, seg2, tmp;
  cv::vector<int> matched(frame1.rows);
  for(int y(0),y_end(std::min(frame1.rows,frame2.rows)-y_step); y<y_end; y+=y_step)
  {
    // cv::Mat seg1(frame1,cv::Rect(0,y,frame1.cols,y_step));
    // cv::Mat seg2(frame2,cv::Rect(0,y,frame2.cols,y_step));
    // cv::reduce(frame1(cv::Rect(0,y,frame1.cols,y_step)),seg1,0,CV_REDUCE_MAX);
    // cv::reduce(frame2(cv::Rect(0,y,frame2.cols,y_step)),seg2,0,CV_REDUCE_MAX);
    cv::Mat seg1(frame1,cv::Rect(0,y,frame1.cols,1));
    cv::Mat seg2(frame2,cv::Rect(0,y,frame2.cols,1));
    int dx(0);
    double match(0.0), max_match(0.0), x_match(0);
    for(int x(0),x_end(frame1.cols); x<x_end; x+=1)
    {
      dx= frame1.cols-x;
      cv::bitwise_and(seg1(cv::Rect(x,0,dx,1)), seg2(cv::Rect(0,0,dx,1)), tmp);
      match= cv::sum(tmp)[0];
      if(match>max_match)  {x_match=x; max_match=match;}
    }
    if(max_match>th_match)
      for(int y2(y);y2<y+y_step;++y2)  matched[y2]= x_match;
    else
      for(int y2(y);y2<y+y_step;++y2)  matched[y2]= -1;
    std::cerr<<" "<<matched[y];
  }
  std::cerr<<std::endl;

  cv::Mat frame1c,frame2c;
  cv::cvtColor(frame1, frame1c, CV_GRAY2BGR);
  cv::cvtColor(frame2, frame2c, CV_GRAY2BGR);
  frame1c/=2;
  frame2c/=2;
  for(int y(0),y_end(matched.size()); y<y_end; ++y)
  {
    int dx= matched[y];
    if(dx>=0)
    {
      for(int x(dx),x_end(frame1.cols); x<x_end; ++x)
      {
        // frame1c.at<cv::Vec3b>(y,x)[0]+= x/2;
        if(frame1.at<unsigned char>(y,x)>10 && frame2.at<unsigned char>(y,x-dx)>10)
        {
          frame1c.at<cv::Vec3b>(y,x)+= cv::Vec3b(x%128,0,128-(x%128));
          frame2c.at<cv::Vec3b>(y,x-dx)+= cv::Vec3b(x%128,0,128-(x%128));
          // frame1.at<unsigned char>(y,x)=255;
          // frame2.at<unsigned char>(y,x-dx)=255;
        }
      }
    }
  }
  frame1c.copyTo(frame1);
  frame2c.copyTo(frame2);

  // cv::Mat disparity;
  // stereo(frame1, frame2, disparity, /*disptype=*/CV_32FC1);
  // return disparity;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture vin1("sample/vout2l.avi");
  cv::VideoCapture vin2("sample/vout2r.avi");
  if(!vin1.isOpened() || !vin2.isOpened())
  {
    std::cerr<<"failed to open!"<<std::endl;
    return -1;
  }
  std::cerr<<"video files opened"<<std::endl;

  cv::namedWindow("video1",1);
  cv::namedWindow("video2",1);
  cv::namedWindow("stereo_flow",1);
  cv::Mat frame1, frame2;

  int n_disp(16*8), w_size(5);
  cv::StereoBM stereo(cv::StereoBM::BASIC_PRESET, /*ndisparities=*/n_disp, /*SADWindowSize=*/w_size);

  bool running(true);
  for(;;)
  {
    if(running)
    {
      bool res1(vin1.read(frame1)), res2(vin2.read(frame2));
      if(!res1 || !res2)
      {
        std::cerr<<"video reached the end (looped)"<<std::endl;
        vin1.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
        vin2.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
        continue;
      }

      cv::cvtColor(frame1, frame1, CV_BGR2GRAY);
      cv::cvtColor(frame2, frame2, CV_BGR2GRAY);
      // cv::Mat disparity= FlowStereo(/*we=*/2, /*wd=*/3, frame1, frame2, stereo);
      FlowStereo2(/*we=*/2, /*wd=*/3, frame1, frame2);
      // cv::normalize(disparity, disparity, 0, 255, CV_MINMAX, CV_8U);
      // cv::imshow("stereo_flow", disparity);

      cv::imshow("video1", frame1);
      cv::imshow("video2", frame2);
    }
    else
    {
      usleep(100*1000);
    }
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c==' ') running= !running;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
