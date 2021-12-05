//-------------------------------------------------------------------------------------------
/*! \file    iterator.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.15, 2015

    g++ -I -Wall iterator.cpp -o iterator.out -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // only for medianBlur
#include <iostream>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
  // return ros::Time::now().toSec();
}
//-------------------------------------------------------------------------------------------

cv::Mat ModImg1(const cv::Mat &img)
{
  cv::Mat res(img.rows, img.cols,CV_8UC3);
  cv::MatConstIterator_<cv::Vec3b> itr= img.begin<cv::Vec3b>();
  cv::MatConstIterator_<cv::Vec3b> itr_end= img.end<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_res= res.begin<cv::Vec3b>();
  for(; itr!=itr_end; ++itr,++itr_res)
  {
    *itr_res= (*itr) * 1.5;
  }
  return res;
}
//-------------------------------------------------------------------------------------------

cv::Mat ModImg2(const cv::Mat &img)
{
  cv::Mat img2;
  cv::medianBlur(img, img2, 3);

  cv::Mat res(img.rows, img.cols,CV_8UC3);
  cv::MatIterator_<cv::Vec3b> itr= img2.begin<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_end= img2.end<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_res= res.begin<cv::Vec3b>();
  for(int i(0); itr!=itr_end; ++itr,++itr_res,++i)
  {
// std::cerr<<" "<<i;
    *itr_res= (*itr) * 0.5;
  }
  return res;
}
//-------------------------------------------------------------------------------------------

cv::Mat ModImg3(const cv::Mat &img)
{
  cv::Mat res(img.rows, img.cols,CV_32FC3);
  cv::MatConstIterator_<cv::Vec3b> itr= img.begin<cv::Vec3b>();
  cv::MatConstIterator_<cv::Vec3b> itr_end= img.end<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3f> itr_res= res.begin<cv::Vec3f>();
  for(; itr!=itr_end; ++itr,++itr_res)
  {
    (*itr_res)[0]= float((*itr)[0]) / 255.0;
    (*itr_res)[1]= float((*itr)[1]) / 128.0;
    (*itr_res)[2]= float((*itr)[2]) / 255.0;
  }
  // std::cerr<<res<<std::endl;
  return res;
}
//-------------------------------------------------------------------------------------------

cv::Mat ModImg4(const cv::Mat &img)
{
  cv::Mat res(img.rows, img.cols,CV_8UC3);
  cv::MatConstIterator_<cv::Vec3b> itr= img.begin<cv::Vec3b>();
  // cv::MatConstIterator_<cv::Vec3b> itr_end= img.end<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_res= res.begin<cv::Vec3b>();
  for(int r(0); r<img.rows; ++r)
    for(int c(0); c<img.cols; ++c,++itr,++itr_res)
    {
      (*itr_res)[0]= (*itr)[0] * float(r)/float(img.rows);
      (*itr_res)[1]= (*itr)[1] * float(c)/float(img.cols);
      (*itr_res)[2]= (*itr)[2];
    }
//   for(int c(0); c<img.cols; ++c)
//     for(int r(0); r<img.rows; ++r,++itr,++itr_res)
//     {
//       (*itr_res)[0]= (*itr)[0] * float(r)/float(img.rows);
//       (*itr_res)[1]= (*itr)[1] * float(c)/float(img.cols);
//       (*itr_res)[2]= (*itr)[2];
//     }
  return res;
}
//-------------------------------------------------------------------------------------------

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

  cv::namedWindow("camera",1);
  cv::Mat frame, disp_img;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    double t_start= GetCurrentTime();
    // disp_img= ModImg1(frame);
    // disp_img= ModImg2(frame);
    // disp_img= ModImg3(frame);
    disp_img= ModImg4(frame);
    std::cerr<<"Computation time: "<<GetCurrentTime()-t_start<<std::endl;

    cv::imshow("camera", disp_img);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
