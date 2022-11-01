//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_segm.cpp
    \brief   Apply contour detection to an edge feature and filter with normal angle.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.01, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_segm.out ros_rs_segm.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
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
#include "ros_proj_mat.cpp"
#include "ros_rs_normal.cpp"
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"
//-------------------------------------------------------------------------------------------

bool mouse_event_detected(false);
int x_mouse(0), y_mouse(0);
std::string win_mouse("");
void setMouseCallback(const std::string &winname, cv::MouseCallback onMouse, const char *userdata)
{
  cv::setMouseCallback(winname, onMouse, const_cast<char*>(userdata));
}
static void onMouse(int event, int x, int y, int /*flags*/, void* param)
{
  if(event == CV_EVENT_LBUTTONDOWN)
  {
    mouse_event_detected= true;
    x_mouse= x; y_mouse= y;
    win_mouse= std::string(reinterpret_cast<const char*>(param));
  }
}
void ProcMouseEvent(const std::string &win, const cv::Mat &m)
{
  if(mouse_event_detected && win_mouse==win)
  {
    std::cout<<win<<": clicked: ("<<x_mouse<<","<<y_mouse<<"): value= "<<GetPixelVal(m,x_mouse,y_mouse)<<std::endl;
    mouse_event_detected= false;
  }
}
//-------------------------------------------------------------------------------------------

double depth_scale(0.3);
cv::Mat proj_mat;

double canny_threshold1=770.0;
double canny_threshold2=780.0;
int    canny_ksize=5;
int    canny_blur_size=3;
double canny_blur_std=1.5;

int    laplacian_ksize=3;
double laplacian_scale=4.0;
double laplacian_delta=0.0;
int    laplacian_blur_size=3;
double laplacian_blur_std=1.5;

int    sobel_ksize=3;
double sobel_scale=3.8;
double sobel_delta=0.0;
int    sobel_blur_size=3;
double sobel_blur_std=1.5;

int edge_kind(2);  //0:canny,1:laplacian,2:sobel
int edge_threshold(100);

int wsize(5);
int cd2ntype(cd2ntSimple);
float resize_ratio(0.5);

float beta_min(0.001), beta_max(0.25);

double dim_image=0.5;
double dim_edge_bin=0.3;
double dim_fbeta=0.3;
double dim_and= 0.7;

bool quit_at_cap_err(false);


void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }
  cv::Mat canny, laplacian, sobel;

  canny= GetCanny(frame_rgb,
    canny_threshold1, canny_threshold2, canny_ksize,
    canny_blur_size, canny_blur_std,
    /*is_depth=*/false);
  laplacian= GetLaplacian(frame_rgb,
    laplacian_ksize, laplacian_scale, laplacian_delta,
    laplacian_blur_size, laplacian_blur_std,
    /*is_depth=*/false);
  sobel= GetSobel(frame_rgb,
    sobel_ksize, sobel_scale, sobel_delta,
    sobel_blur_size, sobel_blur_std,
    /*is_depth=*/false);

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


  cv::Mat normal_img, cloud_img;
  DepthImgToNormalImg(
    frame_depth, proj_mat,
    normal_img, cloud_img, wsize, resize_ratio, /*type=*/TCD2NType(cd2ntype));  // cd2ntSimple,cd2ntRobust

  cv::Mat alpha_beta_img;
  PolarizeNormalImg(normal_img, alpha_beta_img);

  cv::Mat filtered_beta;
  cv::inRange(alpha_beta_img, cv::Scalar(-1.f,beta_min,-1.f), cv::Scalar(1.f,beta_max,1.f), filtered_beta);


  cv::Mat edge_binary_and_normal_beta;
  cv::bitwise_and(edge_binary,filtered_beta, edge_binary_and_normal_beta);


  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(edge_binary_and_normal_beta, contours, /*CV_RETR_EXTERNAL*/CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  cv::Mat edge_contour,edge_bin_col;
  edge_contour= dim_image*frame_rgb;
  cv::Mat edge_bin_col_decom[3]= {dim_edge_bin*edge_binary,dim_fbeta*filtered_beta,dim_and*edge_binary_and_normal_beta};
  cv::merge(edge_bin_col_decom,3,edge_bin_col);
  edge_contour+= edge_bin_col;
  for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    cv::drawContours(edge_contour, contours, ic, CV_RGB(255,0,255), /*thickness=*/1, /*linetype=*/8);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, CV_GRAY2BGR);
  cv::imshow("input_depth", img_depth_disp);
  ProcMouseEvent("input_depth", frame_depth);

  cv::imshow("input_rgb", frame_rgb);
  ProcMouseEvent("input_rgb", frame_rgb);

  cv::imshow("canny", canny);
  ProcMouseEvent("canny", canny);
  cv::imshow("laplacian", laplacian);
  ProcMouseEvent("laplacian", laplacian);
  cv::imshow("sobel", sobel);
  ProcMouseEvent("sobel", sobel);

  cv::imshow("edge_binary", edge_binary);
  ProcMouseEvent("edge_binary", edge_binary);

  cv::imshow("normal", normal_img);
  ProcMouseEvent("normal", normal_img);

  cv::imshow("alpha_beta", alpha_beta_img);
  ProcMouseEvent("alpha_beta", alpha_beta_img);

  cv::imshow("filtered_beta", filtered_beta);
  ProcMouseEvent("filtered_beta", filtered_beta);

  cv::imshow("edge_contour", edge_contour);
  ProcMouseEvent("edge_contour", edge_contour);

  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img1_topic("/camera/aligned_depth_to_color/image_raw");
  std::string img2_topic("/camera/color/image_raw");
  if(argc>1)  img1_topic= argv[1];
  if(argc>2)  img2_topic= argv[2];
  ros::init(argc, argv, "ros_rs_edge_cmp");
  ros::NodeHandle node("~");
  std::string encoding1= GetImageEncoding(img1_topic, node, /*convert_cv=*/true);
  std::string encoding2= GetImageEncoding(img2_topic, node, /*convert_cv=*/true);
  if(encoding1!="16UC1")
  {
    std::cerr<<"WARNING: We assume img1 as a depth image topic, but is "<<encoding1<<std::endl;
  }

  // cv::Mat proj_mat;
  std::string img_info_topic(img1_topic), frame_id, ltopic("image_raw");
  img_info_topic.replace(img_info_topic.find(ltopic),ltopic.length(),"camera_info");
  GetCameraProjectionMatrix(img_info_topic, frame_id, proj_mat);

  cv::namedWindow("input_depth",1);
  setMouseCallback("input_depth", onMouse, "input_depth");
  cv::namedWindow("input_rgb",1);
  setMouseCallback("input_rgb", onMouse, "input_rgb");

  cv::namedWindow("canny",1);
  setMouseCallback("canny", onMouse, "canny");
  CreateTrackbar<double>("threshold1", "canny", &canny_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny", &canny_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("ksize",      "canny", &canny_ksize, 3, 7, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size",  "canny", &canny_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std",   "canny", &canny_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("laplacian",1);
  setMouseCallback("laplacian", onMouse, "laplacian");
  CreateTrackbar<int>   ("ksize",    "laplacian", &laplacian_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "laplacian", &laplacian_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "laplacian", &laplacian_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","laplacian", &laplacian_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian", &laplacian_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("sobel",1);
  setMouseCallback("sobel", onMouse, "sobel");
  CreateTrackbar<int>   ("ksize",    "sobel", &sobel_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale",    "sobel", &sobel_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta",    "sobel", &sobel_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("blur_size","sobel", &sobel_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel", &sobel_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::namedWindow("edge_binary",1);
  setMouseCallback("edge_binary", onMouse, "edge_binary");
  CreateTrackbar<int>   ("edge_kind",      "edge_binary", &edge_kind, 0, 2, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>   ("edge_threshold", "edge_binary", &edge_threshold, 0, 255, 1,  &TrackbarPrintOnTrack);

  cv::namedWindow("normal",1);
  setMouseCallback("normal", onMouse, "normal");
  CreateTrackbar<int>("wsize", "normal", &wsize, 1, 15, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("cd2ntype", "normal", &cd2ntype, 0, 1, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_ratio", "normal", &resize_ratio, 0.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);

  cv::namedWindow("alpha_beta",1);
  setMouseCallback("alpha_beta", onMouse, "alpha_beta");

  cv::namedWindow("filtered_beta",1);
  setMouseCallback("filtered_beta", onMouse, "filtered_beta");
  CreateTrackbar<float>("beta_min", "filtered_beta", &beta_min, -1.0f, 1.0f, 0.01,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("beta_max", "filtered_beta", &beta_max, -1.0f, 1.0f, 0.01,  &TrackbarPrintOnTrack);

  cv::namedWindow("edge_contour",1);
  setMouseCallback("edge_contour", onMouse, "edge_contour");
  CreateTrackbar<double>("dim_image", "edge_contour", &dim_image, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("dim_edge_bin", "edge_contour", &dim_edge_bin, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("dim_fbeta", "edge_contour", &dim_fbeta, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("dim_and", "edge_contour", &dim_and, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
