//-------------------------------------------------------------------------------------------
/*! \file    calib_drawscale.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.10, 2020

g++ -g -Wall -O2 -o calib_drawscale.out calib_drawscale.cpp -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui
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
#define LIBRARY
#include "calib_man.cpp"
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

// Project points 3D onto a rectified image.
void ProjectPointsToRectifiedImg(const cv::Mat &points3d, const cv::Mat &P, cv::Mat &points2d)
{
  assert(points3d.type()==CV_32F);
  cv::Mat P2;
  P.convertTo(P2,points3d.type());
// std::cerr<<"P2="<<P2<<std::endl;
// std::cerr<<"P2'="<<P2(cv::Range(0,3),cv::Range(0,3))<<std::endl;
// std::cerr<<"points3d="<<points3d<<std::endl;

  cv::Mat points2dh= points3d*P2(cv::Range(0,3),cv::Range(0,3)).t();
  cv::Mat p3;
  if(P2.cols>3)  p3= P2.col(3).t();
  for(int r(0),rows(points2dh.rows);r<rows;++r)
  {
    if(P2.cols>3)  points2dh.row(r)+= p3;
    if(points2dh.at<float>(r,2)<0.0)  points2dh.at<float>(r,2)= 0.001;
  }
  // cv::convertPointsFromHomogeneous(points2dh, points2d);
  // points2d= points2d.reshape(1);
  points2d= cv::Mat(points2dh.rows,points2dh.cols,points2dh.type());
  for(int r(0),rows(points2dh.rows);r<rows;++r)
  {
    points2d.at<float>(r,0)= points2dh.at<float>(r,0)/points2dh.at<float>(r,2);
    points2d.at<float>(r,1)= points2dh.at<float>(r,1)/points2dh.at<float>(r,2);
  }
}
//-------------------------------------------------------------------------------------------

void DrawScale(cv::Mat &img, const cv::Mat &P,
  const float &x0, const float &y, const float &z,  // start point
  const float &x1,  // end point
  const float &tics, const float &mtics,  // interval
  int len_tics, int len_mtics,  // length of tics
  int label0, int label_step
)
{
  cv::Mat tics_points3d(int((x1-x0)/tics)+1,3,CV_32F);
  cv::Mat mtics_points3d(int((x1-x0)/mtics)+1,3,CV_32F);
  cv::Mat tics_points2d;
  cv::Mat mtics_points2d;

  for(int i(0);i<tics_points3d.rows;++i)
  {
    tics_points3d.at<float>(i,0)= tics*(float)i + x0;
    tics_points3d.at<float>(i,1)= y;
    tics_points3d.at<float>(i,2)= z;
  }
  ProjectPointsToRectifiedImg(tics_points3d, P, tics_points2d);

  for(int i(0);i<mtics_points3d.rows;++i)
  {
    mtics_points3d.at<float>(i,0)= mtics*(float)i + x0;
    mtics_points3d.at<float>(i,1)= y;
    mtics_points3d.at<float>(i,2)= z;
  }
  ProjectPointsToRectifiedImg(mtics_points3d, P, mtics_points2d);

  #define pointat(m,i)  cv::Point(m.at<float>(i,0),m.at<float>(i,1))
  cv::Scalar col(0,0,255);
  // Main axis
  for(int i(1);i<mtics_points2d.rows;++i)
    cv::line(img, pointat(mtics_points2d,i), pointat(mtics_points2d,i-1), col, 1);
  // Small tics
  for(int i(0);i<mtics_points2d.rows;++i)
    cv::line(img, pointat(mtics_points2d,i), pointat(mtics_points2d,i)+cv::Point(0,len_mtics), col, 1);
  // Main tics
  for(int i(0);i<tics_points2d.rows;++i)
    cv::line(img, pointat(tics_points2d,i), pointat(tics_points2d,i)+cv::Point(0,len_tics), col, 2);
  // Labels
  for(int i(0);i<tics_points2d.rows;++i)
  {
    std::stringstream ss;
    ss<<label0+label_step*i;
    cv::putText(img, ss.str(), pointat(tics_points2d,i)+cv::Point(-10,-2*len_tics), cv::FONT_HERSHEY_SIMPLEX, 0.8, col, 1, CV_AA);
  }
  #undef pointat
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::FileStorage calib((argc>1)?(argv[1]):"calib/camera.yaml", cv::FileStorage::READ);
  if(!calib.isOpened())
  {
    std::cerr<<"Failed to open calibration file"<<std::endl;
    return -1;
  }
  TCapture cap;
  if(!cap.Open(((argc>2)?(argv[2]):"0"), /*width=*/((argc>3)?atoi(argv[3]):0), /*height=*/((argc>4)?atoi(argv[4]):0)))  return -1;
  std::string record= (argc>5)?(argv[5]):"";
  int record_fps= (argc>6)?atoi(argv[6]):12;

  int height, width;
  double Alpha(1.0);
  cv::Mat K,D,R(cv::Mat::eye(3, 3, CV_64F));
  if(!calib["Alpha"].isNone())  calib["Alpha"] >> Alpha;
  calib["camera_matrix"] >> K;
  calib["distortion_coefficients"] >> D;
  calib["image_width"] >> width;
  calib["image_height"] >> height;
  TCameraRectifier cam_rectifier;
  cv::Size size_in(width,height), size_out(width,height);
  cam_rectifier.Setup(K, D, R, size_in, Alpha, size_out);
  cv::Mat P= cv::getOptimalNewCameraMatrix(K, D, size_in, Alpha, size_out);

  cv::namedWindow("camera",1);
  cv::namedWindow("trackbar",1);

  float x0(-0.154),y(-0.031),z(0.375);
  float x1(0.162);
  float tics(0.1), mtics(0.01);
  int len_tics(10), len_mtics(5);
  int label0(0), label_step(10);
  if(!calib["scale"].isNone())
  {
    cv::FileNode fn= calib["scale"];
    if(!fn["x0"        ].isNone())  fn["x0"        ] >> x0        ;
    if(!fn["y"         ].isNone())  fn["y"         ] >> y         ;
    if(!fn["z"         ].isNone())  fn["z"         ] >> z         ;
    if(!fn["x1"        ].isNone())  fn["x1"        ] >> x1        ;
    if(!fn["tics"      ].isNone())  fn["tics"      ] >> tics      ;
    if(!fn["mtics"     ].isNone())  fn["mtics"     ] >> mtics     ;
    if(!fn["len_tics"  ].isNone())  fn["len_tics"  ] >> len_tics  ;
    if(!fn["len_mtics" ].isNone())  fn["len_mtics" ] >> len_mtics ;
    if(!fn["label0"    ].isNone())  fn["label0"    ] >> label0    ;
    if(!fn["label_step"].isNone())  fn["label_step"] >> label_step;
  }
  CreateTrackbar<float>("x0", "trackbar", &x0, -0.5, 0.5, 0.001);
  CreateTrackbar<float>("y", "trackbar", &y, -0.2, 0.2, 0.001);
  CreateTrackbar<float>("z", "trackbar", &z, 0.1, 1.0, 0.001);
  CreateTrackbar<float>("x1", "trackbar", &x1, -0.5, 0.5, 0.001);

  // Open the video recorder.
  cv::VideoWriter vout;
  if(record!="")
  {
    std::string file_name= record;
    // int codec= CV_FOURCC('P','I','M','1');  // mpeg1video
    // int codec= CV_FOURCC('X','2','6','4');  // x264?
    int codec= CV_FOURCC('m','p','4','v');  // mpeg4 (Simple Profile)
    // int codec= CV_FOURCC('X','V','I','D');
    vout.open(file_name, codec, record_fps, cv::Size(width, height), true);
    if (!vout.isOpened())
    {
      std::cout<<"Failed to open the output video: "<<file_name<<std::endl;
      return -1;
    }
    std::cout<<"Output video: "<<file_name<<std::endl;
  }

  cv::Mat frame;
  for(int i(0),i_saved(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(record=="" && cap.WaitReopen()) continue;
      else break;
    }
    cam_rectifier.Rectify(frame, /*border=*/cv::Scalar(0,0,0));
    DrawScale(frame, P, x0, y, z, x1, tics, mtics, len_tics, len_mtics, label0, label_step);
    cv::imshow("camera", frame);
    cv::imshow("trackbar", cv::Mat(1,1000,CV_64F));

    // Record the video.
    if(record!="")  vout<<frame;

    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    else if(c=='p')
    {
      std::cerr<<"scale: "<<x0<<std::endl;
      std::cerr<<"  x0: "<<x0<<std::endl;
      std::cerr<<"  y: " <<y<<std::endl;
      std::cerr<<"  z: " <<z<<std::endl;
      std::cerr<<"  x1: "<<x1<<std::endl;
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
//-------------------------------------------------------------------------------------------
