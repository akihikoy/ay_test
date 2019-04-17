//-------------------------------------------------------------------------------------------
/*! \file    emph_diff.cpp
    \brief   Emphasized difference image.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.15, 2015

    g++ -I -Wall emph_diff.cpp -o emph_diff.out -lopencv_core -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

inline void EmphasizeDiff(cv::Vec3f &diff)
{
  for(int d(0);d<3;++d)
  {
    // diff[d]= diff[d]/255.0;
    diff[d]= std::fabs(diff[d]/255.0);
    // if(diff[d]<0.0)  diff[d]= 0.0;
    // if(diff[d]>0.01)  diff[d]= diff[d]*10.0;
    // if(diff[d]>0.01)  diff[d]= std::atan(diff[d]*10.0);
    diff[d]= diff[d]*10.0;
  }
}

cv::Mat GetEmphasizedDiff(const cv::Mat &img_old, const cv::Mat &img)
{
  cv::Mat img1, img2;
  cv::Mat diff(img.rows, img.cols,CV_32FC3);
  cv::medianBlur(img_old, img1, 3);
  cv::medianBlur(img, img2, 3);

// std::cerr<<"diff "<<diff.rows<<","<<diff.cols<<std::endl;
// std::cerr<<"img1 "<<img1.rows<<","<<img1.cols<<std::endl;
// std::cerr<<"img2 "<<img2.rows<<","<<img2.cols<<std::endl;
// std::cerr<<" "<<img1.rows*img1.cols<<std::endl;
  cv::MatIterator_<cv::Vec3b> itr_2= img2.begin<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_2_end= img2.end<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3b> itr_1= img1.begin<cv::Vec3b>();
  cv::MatIterator_<cv::Vec3f> itr_d= diff.begin<cv::Vec3f>();
  for(; itr_2!=itr_2_end; ++itr_2,++itr_1,++itr_d)
  {
    for(int d(0);d<3;++d)
      (*itr_d)[d]= float((*itr_2)[d])-float((*itr_1)[d]);
    EmphasizeDiff(*itr_d);
  }
// std::cerr<<diff<<std::endl;
  return diff;
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

  const int N(1);
  cv::namedWindow("camera",1);
  cv::Mat frame, frame_old[N], disp_img;
  for(int i(0);;++i)
  {
    int n((i+1)%N);
    cap >> frame; // get a new frame from camera
    if(frame_old[n].empty())  frame.copyTo(frame_old[n]);

    disp_img= GetEmphasizedDiff(frame_old[n], frame);

    cv::imshow("camera", disp_img);
    frame.copyTo(frame_old[i%N]);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
//-------------------------------------------------------------------------------------------
