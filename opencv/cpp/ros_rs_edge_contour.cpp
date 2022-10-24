//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_edge_contour.cpp
    \brief   Apply contour detection to an edge feature.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.25, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_edge_contour.out ros_rs_edge_contour.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
#define LIBRARY
#include "cv2-edge_cmp.cpp"
#include "ros_capture.cpp"
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

bool is_depth(false);
double depth_scale(0.3);

double canny_threshold1=100.0;
double canny_threshold2=200.0;
int    canny_ksize=5;
int    canny_blur_size=3;
double canny_blur_std=1.5;

int    laplacian_ksize=5;
double laplacian_scale=2.0;
double laplacian_delta=0.0;
int    laplacian_blur_size=1;
double laplacian_blur_std=1.5;

int    sobel_ksize=3;
double sobel_scale=6.0;
double sobel_delta=0.0;
int    sobel_blur_size=1;
double sobel_blur_std=1.5;

int edge_kind(2);  //0:canny,1:laplacian,2:sobel
int edge_threshold(100);

double dim_image=0.5;
double dim_edge_bin=0.7;

bool quit_at_cap_err(false), disp_sum(false);
double sumscale(1.0e-6);

void CVCallback(const cv::Mat &frame)
{
  if(frame.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }
  cv::Mat canny, laplacian, sobel;

  canny= GetCanny(frame,
    canny_threshold1, canny_threshold2, canny_ksize,
    canny_blur_size, canny_blur_std,
    is_depth);
  laplacian= GetLaplacian(frame,
    laplacian_ksize, laplacian_scale, laplacian_delta,
    laplacian_blur_size, laplacian_blur_std,
    is_depth);
  sobel= GetSobel(frame,
    sobel_ksize, sobel_scale, sobel_delta,
    sobel_blur_size, sobel_blur_std,
    is_depth);

  cv::Mat edge_binary;
  if     (edge_kind==0)
    cv::threshold(canny,edge_binary,edge_threshold,255,cv::THRESH_BINARY);
  else if(edge_kind==1)
    cv::threshold(laplacian,edge_binary,edge_threshold,255,cv::THRESH_BINARY);
  else if(edge_kind==2)
  {
    cv::Mat gray;
    cv::cvtColor(sobel, gray, CV_BGR2GRAY);
    cv::threshold(gray,edge_binary,edge_threshold,255,cv::THRESH_BINARY);
  }

  cv::Mat img_disp;
  if(!is_depth)
    img_disp= frame;
  else
  {
    img_disp= frame*depth_scale;
    img_disp.convertTo(img_disp, CV_8U);
    cv::cvtColor(img_disp, img_disp, CV_GRAY2BGR);
  }

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(edge_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  cv::Mat edge_contour,edge_bin_col;
  edge_contour= dim_image*img_disp;
  cv::Mat edge_bin_col_decom[3]= {128.0*edge_binary,128.0*edge_binary,128.0*edge_binary+128.0*edge_binary}, mask_objectsc;
  cv::merge(edge_bin_col_decom,3,edge_bin_col);
  edge_contour+= dim_edge_bin*edge_bin_col;
  for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    cv::drawContours(edge_contour, contours, ic, CV_RGB(255,0,255), /*thickness=*/1, /*linetype=*/8);

  cv::imshow("input", img_disp);

  cv::imshow("canny", canny);
  cv::imshow("laplacian", laplacian);
  cv::imshow("sobel", sobel);

  cv::imshow("edge_binary", edge_binary);
  cv::imshow("edge_contour", edge_contour);

  if(disp_sum)
    std::cout<<"canny, laplacian, sobel: "<<cv::sum(canny)*sumscale<<", "<<cv::sum(laplacian)*sumscale<<", "<<cv::sum(sobel)*sumscale<<std::endl;

  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
  if(c==' ')  disp_sum=!disp_sum;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/color/image_raw"), encoding;
  if(argc>1)  img_topic= argv[1];
  ros::init(argc, argv, "ros_rs_edge_cmp");
  ros::NodeHandle node("~");
  encoding= GetImageEncoding(img_topic, node, /*convert_cv=*/true);
  if(encoding=="16UC1")  is_depth= true;

  cv::namedWindow("input",1);

  cv::namedWindow("canny",1);
  CreateTrackbar<double>("threshold1", "canny", &canny_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny", &canny_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("ksize",      "canny", &canny_ksize, 3, 7, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size",  "canny", &canny_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std",   "canny", &canny_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("laplacian",1);
  CreateTrackbar<int>   ("ksize",    "laplacian", &laplacian_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "laplacian", &laplacian_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "laplacian", &laplacian_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","laplacian", &laplacian_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian", &laplacian_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("sobel",1);
  CreateTrackbar<int>   ("ksize",    "sobel", &sobel_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "sobel", &sobel_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "sobel", &sobel_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","sobel", &sobel_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel", &sobel_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("edge_binary",1);
  CreateTrackbar<int>   ("edge_kind",      "edge_binary", &edge_kind, 0, 2, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("edge_threshold", "edge_binary", &edge_threshold, 0, 255, 1,  &TrackbarPrintOnTrack);

  cv::namedWindow("edge_contour",1);
  CreateTrackbar<double>("dim_image", "edge_contour", &dim_image, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("dim_edge_bin", "edge_contour", &dim_edge_bin, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img_topic, encoding, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
