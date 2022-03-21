//-------------------------------------------------------------------------------------------
/*! \file    cv2-video_fps.cpp
    \brief   Get FPS of a video file.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Mar.21, 2022

g++ -g -Wall -O2 -o cv2-video_fps.out cv2-video_fps.cpp -lopencv_core -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------
inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture vin("sample/vout2l.avi");
  if(argc==2)
  {
    vin.release();
    vin.open(argv[1]);
  }
  if(!vin.isOpened())  // check if we succeeded
  {
    std::cerr<<"failed to open!"<<std::endl;
    return -1;
  }
  std::cerr<<"video file opened"<<std::endl;
  vin.set(CV_CAP_PROP_FPS, 15);  // TEST:Is setting FPS affects get(FPS)?
                                 // Result: No, set(FPS) does not affects neither get(FPS) or actual vin reading.

  std::cerr<<"FPS from vin.get(CAP_PROP_FPS): "<<vin.get(CV_CAP_PROP_FPS)<<std::endl;

  cv::namedWindow("video",1);
  cv::Mat frame;
  vin.read(frame);  // dummy
  int num_frame_read(0);
  double t_start(GetCurrentTime());
  for(;;)
  {
    if(!vin.read(frame))
    {
      std::cerr<<"video reached the end"<<std::endl;
      break;
    }
    ++num_frame_read;

    cv::imshow("video", frame);
    int c(cv::waitKey(1));
    if(c=='\x1b'||c=='q')  break;
    if(num_frame_read>=120)  break;
  }

  std::cerr<<"FPS of vin reading: "<<num_frame_read/(GetCurrentTime()-t_start)<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
