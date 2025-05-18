//-------------------------------------------------------------------------------------------
/*! \file    calib_man.cpp
    \brief   Manual calibration adjuster.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o calib_man.out calib_man.cpp -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -I/usr/include/opencv4
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

//-------------------------------------------------------------------------------------------
#ifndef LIBRARY
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

double Alpha;
cv::Mat K, D;
int height, width;

bool ParamChanged(false);

template<typename T>
void OnTrack(const TExtendedTrackbarInfo<T> &info, void*)
{
  std::cerr<<"Changed: "<<info.Name<<": "<<info.Value<<std::endl;
  ParamChanged= true;
}

bool LoadCalib(const std::string &calib_filename)
{
  cv::FileStorage calib(calib_filename, cv::FileStorage::READ);
  if(!calib.isOpened())
  {
    std::cerr<<"Failed to open calibration file: "<<calib_filename<<std::endl;
    return false;
  }
  calib["camera_matrix"] >> K;
  calib["distortion_coefficients"] >> D;
  calib["image_width"] >> width;
  calib["image_height"] >> height;
  return true;
}

int main(int argc, char**argv)
{
  std::string calib_filename((argc>1)?(argv[1]):"calib/camera.yaml");
  Alpha= (argc>2)?atof(argv[2]):1.0;
  TCapture cap;
  if(!cap.Open(((argc>3)?(argv[3]):"0"), /*width=*/((argc>4)?atoi(argv[4]):0), /*height=*/((argc>5)?atoi(argv[5]):0)))  return -1;

  if(!LoadCalib(calib_filename))  return -1;
  cv::Mat R(cv::Mat::eye(3, 3, CV_64F));
  TCameraRectifier cam_rectifier;
  cv::Size size_in(width,height), size_out(width,height);
  cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);

  cv::namedWindow("camera",1);

  std::string window("trackbar");
  cv::namedWindow(window,1);

  CreateTrackbar<double>("Alpha", window, &Alpha, 0.0, 2.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("K00", window, &K.at<double>(0,0), 0.0f, 5000.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("K02", window, &K.at<double>(0,2), 0.0f, 5000.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("K11", window, &K.at<double>(1,1), 0.0f, 5000.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("K12", window, &K.at<double>(1,2), 0.0f, 5000.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("D0", window, &D.at<double>(0,0), -10.0, 10.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("D1", window, &D.at<double>(0,1), -10.0, 10.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("D2", window, &D.at<double>(0,2), -10.0, 10.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("D3", window, &D.at<double>(0,3), -10.0, 10.0, 1e-5,  &OnTrack);
  CreateTrackbar<double>("D4", window, &D.at<double>(0,4), -10.0, 10.0, 1e-5,  &OnTrack);

  cv::Mat frame;
  bool with_rectify(true);
  for(int i(0),i_saved(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    if(ParamChanged)
    {
      cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);
      ParamChanged= false;
    }
    if(with_rectify)  cam_rectifier.Rectify(frame, /*border=*/cv::Scalar(0,0,0));
    cv::imshow("camera", frame);
    cv::imshow("trackbar", cv::Mat(1,1000,CV_64F));
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    else if(c==' ')
      with_rectify= !with_rectify;
    else if(c=='p')
    {
      cv::FileStorage fs("/dev/stdout", cv::FileStorage::WRITE);
      fs<<"camera_matrix"<<K;
      fs<<"distortion_coefficients"<<D;
      fs.release();
    }
    else if(c=='l')
    {
      if(!LoadCalib(calib_filename))  return -1;
      cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);
    }
    else if(c=='s')
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
