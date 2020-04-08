//-------------------------------------------------------------------------------------------
/*! \file    timer_video_rec.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.08, 2020

g++ -g -Wall -O3 -o timer_video_rec.out timer_video_rec.cpp -lopencv_core -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <unistd.h>  // usleep
#include <ctime>  // clock_gettime
#include <sys/time.h>  // gettimeofday
#include <unistd.h>
#include "cap_open.h"
#include "cv2-videoout2.h"
//-------------------------------------------------------------------------------------------

struct TTime
{
  long Sec;  // Seconds
  long NSec;  // Nano-seconds
  TTime() : Sec(0), NSec(0)  {}
  TTime(const long &s, const long &ns) : Sec(s), NSec(ns)  {}
  double ToSec()  {return static_cast<double>(Sec) + static_cast<double>(NSec)*1.0e-9l;}
  void Normalize()
    {
      long nsec2= NSec % 1000000000L;
      long sec2= Sec + NSec / 1000000000L;
      if (nsec2 < 0)
      {
        nsec2+= 1000000000L;
        --sec2;
      }
      Sec= sec2;
      NSec= nsec2;
    }
};
inline TTime operator+(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec+= rhs.Sec;
  res.NSec+= rhs.NSec;
  res.Normalize();
  return res;
}
inline TTime operator-(const TTime &lhs, const TTime &rhs)
{
  TTime res(lhs);
  res.Sec-= rhs.Sec;
  res.NSec-= rhs.NSec;
  res.Normalize();
  return res;
}
#define HAS_CLOCK_GETTIME (_POSIX_C_SOURCE >= 199309L)
inline TTime GetCurrentTime(void)
{
#if HAS_CLOCK_GETTIME
  timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  TTime res;
  res.Sec= start.tv_sec;
  res.NSec= start.tv_nsec;
  return res;
#else
  struct timeval time;
  gettimeofday (&time, NULL);
  TTime res;
  res.Sec= time.tv_sec;
  res.NSec= time.tv_usec*1000L;
  return res;
#endif
}


int main(int argc, char**argv)
{
  TTime t_current= GetCurrentTime();
  TCapture cap;
  // Read the command line options and open the camera device.
  int iarg(1);
  long t_start_sec= (argc>iarg)?atoi(argv[iarg]):0; ++iarg;
  long t_start_nsec= 0;
  int FPS= (argc>iarg)?atoi(argv[iarg]):15; ++iarg;
  int duration= (argc>iarg)?atoi(argv[iarg]):10; ++iarg;
  if(!cap.Open(((argc>iarg)?(argv[iarg]):"0"),
              /*width=*/((argc>iarg+1)?atoi(argv[iarg+1]):0),
              /*height=*/((argc>iarg+2)?atoi(argv[iarg+2]):0)))  return -1;
  iarg+= 3;
  bool disp= (argc>iarg)?atoi(argv[iarg]):1; ++iarg;
  bool record= (argc>iarg)?atoi(argv[iarg]):0; ++iarg;

  // Command check.
  std::cerr<<"Current time:"<<std::endl;
  std::cerr<<std::setfill('0')<<std::setw(9)<<t_current.Sec<<" "<<std::setfill('0')<<std::setw(9)<<t_current.NSec<<std::endl;
  if(t_start_sec<0)
  {
    std::cerr<<"\nUsage: ./timer_video_rec.out T_START FPS DURATION CAM_ID WIDTH HEIGHT DISP RECORD\n"
      "T_START: Time to start (sec).  Options: absolute Unix time, seconds after current time, 0 (auto).\n"
      "FPS:     FPS to capture.\n"
      "DURATION: Recoding duration (sec).\n"
      "CAM_ID WIDTH HEIGHT:  Camera device ID, image width and height.\n"
      "DISP RECORD:  Displaying the video (0,1), Recording the video (0,1).\n"
      "Showing this help: ./timer_video_rec.out -1\n"
      "Example: ./timer_video_rec.out 5 25 180 1 1920 1080 0 1\n"
      "  -> Recoding 5 sec after, at 25 FPS, for 180 sec, cam: 1, 1920x1080, no display, recording.\n"
      <<std::endl;
    return 0;
  }
  if(t_start_sec==0)  t_start_sec= 2;
  if(t_start_sec<t_current.Sec)
    t_start_sec= t_current.Sec+t_start_sec+(t_current.NSec>500000000L?1:0);
  long interval= 1000000000L/FPS;

  if(disp)  cv::namedWindow("camera",1);
  cv::Mat frame;
  cap>>frame;
  double font_scale(0.8);
  cv::Size text_size;
  int text_baseline;
  {
    std::stringstream ss;
    ss<<std::setfill('0')<<std::setw(9)<<t_current.Sec<<" "<<std::setfill('0')<<std::setw(9)<<t_current.NSec;
    text_size= cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &text_baseline);
  }
  if(disp)
  {
    cv::imshow("camera", frame);
    cv::waitKey(1);
  }

  // Open the video recorder.
  cv::VideoWriter vout;
  if(record)
  {
    std::stringstream ss;
    ss<<"/tmp/vout"<<std::setfill('0')<<std::setw(9)<<t_start_sec<<".avi";
    std::string file_name= ss.str();
    // int codec= CV_FOURCC('P','I','M','1');  // mpeg1video
    // int codec= CV_FOURCC('X','2','6','4');  // x264?
    int codec= CV_FOURCC('m','p','4','v');  // mpeg4 (Simple Profile)
    vout.open(file_name.c_str(), codec, FPS, cv::Size(frame.cols, frame.rows), true);
    if (!vout.isOpened())
    {
      std::cout<<"Failed to open the output video: "<<file_name<<std::endl;
      return -1;
    }
    std::cout<<"Output video: "<<file_name<<std::endl;
  }

  // Wait for t_start.
  TTime t0= TTime(t_start_sec,t_start_nsec);
  std::cerr<<"Recording starts at:"<<std::endl;
  std::cerr<<std::setfill('0')<<std::setw(9)<<t0.Sec<<" "<<std::setfill('0')<<std::setw(9)<<t0.NSec<<std::endl;
  for(;;)
  {
    TTime dt= t0-GetCurrentTime();
    if(dt.ToSec()>0.1)
    {
      cap>>frame;
    }
    else if(dt.ToSec()>0.0)
    {
      usleep(dt.Sec*1000000+dt.NSec/1000);
      break;
    }
    else break;
  }

  for(int i(0);;++i)
  {
    TTime t1= GetCurrentTime();
    std::cerr<<std::setfill('0')<<std::setw(9)<<(t1-t0).Sec<<" "<<std::setfill('0')<<std::setw(9)<<(t1-t0).NSec<<std::endl;

    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    // Print time stamp on the right corner of the image.
    {
      std::stringstream ss;
      ss<<std::setfill('0')<<std::setw(9)<<t1.Sec<<" "<<std::setfill('0')<<std::setw(9)<<t1.NSec;
      cv::putText(frame, ss.str(), cv::Point(frame.cols-text_size.width-5,frame.rows-text_baseline-5), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0,255,0), 1, CV_AA);
    }

    // Record the video.
    if(record)  vout<<frame;

    if(disp)
    {
      cv::imshow("camera", frame);
      char c(cv::waitKey(1));
      if(c=='\x1b'||c=='q') break;
    }

    // Check if duration has passed.
    if((t1-t0).Sec>=duration)  break;

    // Sleep to adjust the rate.
    TTime dt= (t0+TTime(0,(i+1)*interval))-GetCurrentTime();
    if(dt.ToSec()>0.0)  usleep(dt.Sec*1000000+dt.NSec/1000);
    else  std::cerr<<"FPS is larger than the camera capability: "<<dt.ToSec()<<std::endl;
  }

  if(vout.isOpened())  vout.release();

  return 0;
}
//-------------------------------------------------------------------------------------------
