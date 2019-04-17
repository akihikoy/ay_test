//-------------------------------------------------------------------------------------------
/*! \file    cv2-videoread.cpp
    \brief   Read video from files
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.06, 2016

g++ -g -Wall -O2 -o cv2-videoread.out cv2-videoread.cpp -lopencv_core -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>
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

  cv::namedWindow("video",1);
  cv::Mat frame;
  for(;;)
  {
    // vin >> frame; // get a new frame from video
    // if (!frame.data)  // loop video
    if(!vin.read(frame))  // get a new frame from video and loop if necessary
    {
      std::cerr<<"video reached the end (looped)"<<std::endl;
      vin.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
      continue;
    }

    cv::imshow("video", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
