//-------------------------------------------------------------------------------------------
/*! \file    calib_man.cpp
    \brief   Manual calibration adjuster.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o calib_man.out calib_man.cpp -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

struct TCameraRectifier
{
  cv::Mat map1_, map2_;
  void Setup(const cv::Mat &K, const cv::Mat &D, const cv::Mat &R, const cv::Size &size_in, const double &alpha, const cv::Size &size_out);
  void Rectify(cv::Mat &frame, const cv::Scalar& border=cv::Scalar());
};
//-------------------------------------------------------------------------------------------
void TCameraRectifier::Setup(const cv::Mat &K, const cv::Mat &D, const cv::Mat &R, const cv::Size &size_in, const double &alpha, const cv::Size &size_out)
{
  cv::Mat P= cv::getOptimalNewCameraMatrix(K, D, size_in, alpha, size_out);
  cv::initUndistortRectifyMap(K, D, R, P, size_out, CV_16SC2, map1_, map2_);
}
//-------------------------------------------------------------------------------------------

void TCameraRectifier::Rectify(cv::Mat &frame, const cv::Scalar& border)
{
  cv::Mat framer;
  cv::remap(frame, framer, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, border);
  frame= framer;
}
//-------------------------------------------------------------------------------------------

#ifndef LIBRARY
double Alpha;
cv::Mat K, D;

int Alpha_100, K00, K02, K11, K12, D0_1000, D1_1000, D2_1000, D3_1000, D4_1000;
bool ParamChanged(false);

void Map()
{
  #define R100(x)  (x*100.0)
  #define R1000(x)  ((10.0+x)*1000.0)
  Alpha_100= R100(Alpha);
  K00= K.at<double>(0,0);
  K02= K.at<double>(0,2);
  K11= K.at<double>(1,1);
  K12= K.at<double>(1,2);
  D0_1000= R1000(D.at<double>(0,0));
  D1_1000= R1000(D.at<double>(0,1));
  D2_1000= R1000(D.at<double>(0,2));
  D3_1000= R1000(D.at<double>(0,3));
  D4_1000= R1000(D.at<double>(0,4));
  #undef R100
  #undef R1000
}
void InvMap(int,void*)
{
  #define R100(x)  ((double)x*0.01)
  #define R1000(x)  ((double)x*0.001-10.0)
  Alpha= R100(Alpha_100);
  K.at<double>(0,0)= K00;
  K.at<double>(0,2)= K02;
  K.at<double>(1,1)= K11;
  K.at<double>(1,2)= K12;
  D.at<double>(0,0)= R1000(D0_1000);
  D.at<double>(0,1)= R1000(D1_1000);
  D.at<double>(0,2)= R1000(D2_1000);
  D.at<double>(0,3)= R1000(D3_1000);
  D.at<double>(0,4)= R1000(D4_1000);
  #undef R100
  #undef R1000
  ParamChanged= true;
}

int main(int argc, char**argv)
{
  cv::FileStorage calib((argc>1)?(argv[1]):"calib/camera.yaml", cv::FileStorage::READ);
  Alpha= (argc>2)?atof(argv[2]):1.0;
  if(!calib.isOpened())
  {
    std::cerr<<"Failed to open calibration file"<<std::endl;
    return -1;
  }
  TCapture cap;
  if(!cap.Open(((argc>3)?(argv[3]):"0"), /*width=*/((argc>4)?atoi(argv[4]):0), /*height=*/((argc>5)?atoi(argv[5]):0)))  return -1;

  int height, width;
  cv::Mat R(cv::Mat::eye(3, 3, CV_64F));
  calib["camera_matrix"] >> K;
  calib["distortion_coefficients"] >> D;
  calib["image_width"] >> width;
  calib["image_height"] >> height;
  TCameraRectifier cam_rectifier;
  cv::Size size_in(width,height), size_out(width,height);
  cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);

  Map();

  cv::namedWindow("camera",1);

  std::string win("trackbar");
  cv::namedWindow(win,1);
  cv::createTrackbar("Alpha_100", win, &Alpha_100, 200, InvMap);
  cv::createTrackbar("K00", win, &K00, 5000, InvMap);
  cv::createTrackbar("K02", win, &K02, 5000, InvMap);
  cv::createTrackbar("K11", win, &K11, 5000, InvMap);
  cv::createTrackbar("K12", win, &K12, 5000, InvMap);
  cv::createTrackbar("D0_1000", win, &D0_1000, 100000, InvMap);
  cv::createTrackbar("D1_1000", win, &D1_1000, 100000, InvMap);
  cv::createTrackbar("D2_1000", win, &D2_1000, 100000, InvMap);
  cv::createTrackbar("D3_1000", win, &D3_1000, 100000, InvMap);
  cv::createTrackbar("D4_1000", win, &D4_1000, 100000, InvMap);

  cv::Mat frame;
  for(int i(0),i_saved(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    if(ParamChanged)
      cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);
    cam_rectifier.Rectify(frame, /*border=*/cv::Scalar(0,0,0));
    cv::imshow("camera", frame);
    cv::imshow("trackbar", cv::Mat(1,1000,CV_64F));
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    else if(c=='p')
    {
      cv::FileStorage fs("/dev/stdout", cv::FileStorage::WRITE);
      fs<<"camera_matrix"<<K;
      fs<<"distortion_coefficients"<<D;
      fs.release();
    }
    else if(c==' ')
    {
      std::stringstream file_name;
      file_name<<"/tmp/view"<<std::setfill('0')<<std::setw(4)<<(i_saved++)<<".png";
      cv::imwrite(file_name.str(), frame);
      std::cout<<"Saved "<<file_name.str()<<std::endl;
    }
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
