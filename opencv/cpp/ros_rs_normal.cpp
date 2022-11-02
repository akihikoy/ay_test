//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_normal.cpp
    \brief   Convert a depth image to a normal image.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Oct.31, 2022

$ g++ -O2 -g -W -Wall -o ros_rs_normal.out ros_rs_normal.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

inline bool IsValidDepth(int d, int d_max=10000)
{
  if(d>0 && d<=d_max)  return true;
  return false;
}
//-------------------------------------------------------------------------------------------

inline bool IsInvalidDepth(int d, int d_max=10000)
{
  if(d<=0 || d>d_max)  return true;
  return false;
}
//-------------------------------------------------------------------------------------------

// Transform a point on an image to a 3D point.
template<typename t_img_depth>
inline cv::Vec3f ImgPointTo3D(int u, int v, const t_img_depth &depth, const cv::Mat &proj_mat)
{
  const double &Fx= proj_mat.at<double>(0,0);
  const double &Fy= proj_mat.at<double>(1,1);
  const double &Cx= proj_mat.at<double>(0,2);
  const double &Cy= proj_mat.at<double>(1,2);
  const double d= depth * 0.001;
  return cv::Vec3f((u-Cx)/Fx*d, (v-Cy)/Fy*d, d);
}
//-------------------------------------------------------------------------------------------

// Convert a depth_img to a point cloud (2d array of xyz 3d points).
// void DepthImgToPointCloud(const cv::Mat &depth_img, const cv::Mat &proj_mat, cv::Mat &cloud_img)
// {
//   cloud_img.create(depth_img.size(), CV_32FC3);
//   const float Fx= proj_mat.at<double>(0,0);
//   const float Fy= proj_mat.at<double>(1,1);
//   const float Cx= proj_mat.at<double>(0,2);
//   const float Cy= proj_mat.at<double>(1,2);
//   float d;
//   cv::MatConstIterator_<unsigned short> itr_depth= depth_img.begin<unsigned short>();
//   cv::MatIterator_<cv::Vec3f> itr_cloud= cloud_img.begin<cv::Vec3f>();
//   for(int y(0); y<depth_img.rows; ++y)
//     for(int x(0); x<depth_img.cols; ++x,++itr_cloud,++itr_depth)
//     {
//       d= (*itr_depth)*0.001f;
//       if(d>0)
//       {
//         (*itr_cloud)[0]= (x-Cx)/Fx*d;
//         (*itr_cloud)[1]= (y-Cy)/Fy*d;
//         (*itr_cloud)[2]= d;
//       }
//       else
//       {
//         (*itr_cloud)[0]= 0.0f;
//         (*itr_cloud)[1]= 0.0f;
//         (*itr_cloud)[2]= 0.0f;
//       }
//     }
// }
//-------------------------------------------------------------------------------------------

// Convert a depth_img to a point cloud (2d array of xyz 3d points).
void DepthImgToPointCloud(const cv::Mat &depth_img, const cv::Mat &proj_mat, cv::Mat &cloud_img)
{
  const float Fx= proj_mat.at<double>(0,0);
  const float Fy= proj_mat.at<double>(1,1);
  const float Cx= proj_mat.at<double>(0,2);
  const float Cy= proj_mat.at<double>(1,2);
  cv::Mat x_img(depth_img.size(), CV_32FC1), y_img(depth_img.size(), CV_32FC1), z_img(depth_img.size(), CV_32FC1);
  cv::MatIterator_<float> itr_x= x_img.begin<float>();
  cv::MatIterator_<float> itr_y= y_img.begin<float>();
  for(int y(0); y<depth_img.rows; ++y)
    for(int x(0); x<depth_img.cols; ++x,++itr_x,++itr_y)
    {
      (*itr_x)= x;
      (*itr_y)= y;
    }
  depth_img.convertTo(z_img,CV_32FC1);
  z_img*= 0.001f;
  x_img= (x_img-Cx).mul(z_img/Fx);
  y_img= (y_img-Cy).mul(z_img/Fy);
  cv::Mat clout_img_decom[3]= {x_img,y_img,z_img};
  cv::merge(clout_img_decom,3,cloud_img);
// for(int y(0); y<depth_img.rows; ++y)for(int x(0); x<depth_img.cols; ++x)
// if(depth_img.at<unsigned short>(y,x)>0&&depth_img.at<unsigned short>(y,x)<250)std::cerr<<"depth_img("<<x<<","<<y<<")="<<depth_img.at<unsigned short>(y,x)<<std::endl;
// for(int y(0); y<depth_img.rows; ++y)for(int x(0); x<depth_img.cols; ++x)
// if(cloud_img.at<cv::Vec3f>(y,x)(2)>0&&cloud_img.at<cv::Vec3f>(y,x)(2)<0.25)std::cerr<<"cloud_img("<<x<<","<<y<<")="<<cloud_img.at<cv::Vec3f>(y,x)<<std::endl;
}
//-------------------------------------------------------------------------------------------

enum TCD2NType {cd2ntSimple=0, cd2ntRobust};

/* Estimate normal and store it as an image.
    normal_img: Output normal image.
    cloud_img: Image of 3d points.
    wsize: Window size for computing normal (should be odd).  */
void DepthImgToNormalImg(
    const cv::Mat &depth_img, const cv::Mat &proj_mat,
    cv::Mat &normal_img, cv::Mat &cloud_img, int wsize, TCD2NType type=cd2ntSimple)
{
  #define DEPTH(x,y)  depth_img.at<unsigned short>(y,x)
  // #define Pt3D(x,y)  ImgPointTo3D(x,y,DEPTH(x,y),proj_mat)
  #define Pt3D(x,y)  cloud_img.at<cv::Vec3f>(y,x)
  // cv::Mat cloud_img;
  if(cloud_img.empty())  DepthImgToPointCloud(depth_img, proj_mat, cloud_img);
  // if(pcloud_img!=NULL)  *pcloud_img= cloud_img.clone();
  int wsizeh((wsize-1)/2);
  normal_img.create(depth_img.size(), CV_32FC3);
  for(int y(0); y<depth_img.rows; ++y)
  {
    for(int x(0); x<depth_img.cols; ++x)
    {
      bool no_data(false);
      if(x<wsizeh || depth_img.cols-wsizeh<=x || y<wsizeh || depth_img.rows-wsizeh<=y)
        no_data= true;
      else if(IsInvalidDepth(DEPTH(x-wsizeh,y)) || IsInvalidDepth(DEPTH(x+wsizeh,y))
            || IsInvalidDepth(DEPTH(x,y-wsizeh)) || IsInvalidDepth(DEPTH(x,y+wsizeh)))
        no_data= true;
      if(no_data)
      {
        normal_img.at<cv::Vec3f>(y,x)= cv::Vec3f(0.0,0.0,0.0);
        continue;
      }
      cv::Vec3f ax1(0.0,0.0,0.0);
      cv::Vec3f ax2(0.0,0.0,0.0);
      if(type==cd2ntSimple)  // Simple calculation (fast)
      {
        ax1= Pt3D(x+wsizeh,y) - Pt3D(x-wsizeh,y);
        ax2= Pt3D(x,y+wsizeh) - Pt3D(x,y-wsizeh);
      }
      else if(type==cd2ntRobust)  // Complicated calculation (slow, but robust for noise)
      {
        int num1(0), num2(0);
        int xs(x-wsizeh), ys(y-wsizeh);
        for(int fs(1); fs<=wsize; fs+=2)
        {
          for(int i(0); i<wsize-fs; ++i)
            for(int j(0); j<wsize; ++j)
              if(IsValidDepth(DEPTH(xs+i,ys+j)) && IsValidDepth(DEPTH(xs+i+fs,ys+j)))
              {
                ax1+= Pt3D(xs+i+fs,ys+j) - Pt3D(xs+i,ys+j);
                ++num1;
              }
          for(int i(0); i<wsize; ++i)
            for(int j(0); j<wsize-fs; ++j)
              if(IsValidDepth(DEPTH(xs+i,ys+j)) && IsValidDepth(DEPTH(xs+i,ys+j+fs)))
              {
                ax2+= Pt3D(xs+i,ys+j+fs) - Pt3D(xs+i,ys+j);
                ++num2;
              }
        }
        ax1/= (float)num1;
        ax2/= (float)num2;
      }

      // cv::Vec3f normal= ax1.cross(ax2);
      cv::Vec3f normal= ax2.cross(ax1);
      if(cv::norm(normal)>1.0e-6)
      {
        cv::normalize(normal,normal);
        normal_img.at<cv::Vec3f>(y,x)= normal;
      }
      else
      {
        normal_img.at<cv::Vec3f>(y,x)= cv::Vec3f(0.0,0.0,0.0);
      }
    }
  }
  #undef Pt3D
  #undef DEPTH
}
//-------------------------------------------------------------------------------------------

void GetProjMatForResizedImg(const cv::Mat &proj_mat, const double &resize_ratio, cv::Mat &proj_mat_s);

/* Estimate normal and store it as an image (with resizing depth_img).
    normal_img: Output normal image.
    cloud_img: Image of 3d points.
    wsize: Window size for computing normal (should be odd).  */
void DepthImgToNormalImg(
    const cv::Mat &depth_img, const cv::Mat &proj_mat,
    cv::Mat &normal_img, cv::Mat &cloud_img, int wsize, const float &resize_ratio=1.0f, TCD2NType type=cd2ntSimple,
    bool fullsize_cloud_img=true)
{
  if(resize_ratio==1.0f)
  {
    DepthImgToNormalImg(depth_img, proj_mat, normal_img, cloud_img, wsize, type);
    return;
  }
  cv::Mat img_depth_s, proj_mat_s;
  cv::resize(depth_img, img_depth_s, cv::Size(), resize_ratio, resize_ratio, cv::INTER_NEAREST);
  GetProjMatForResizedImg(proj_mat, resize_ratio, proj_mat_s);
  cv::Mat cloud_img_s;
  if(fullsize_cloud_img)
  {
    DepthImgToPointCloud(depth_img, proj_mat, cloud_img);
    cv::resize(cloud_img, cloud_img_s, img_depth_s.size(), 0, 0, cv::INTER_NEAREST);
  }
  cv::Mat normal_img_s;
  DepthImgToNormalImg(img_depth_s, proj_mat_s, normal_img_s, cloud_img_s, wsize, type);
  cv::resize(normal_img_s, normal_img, depth_img.size(), 0, 0, cv::INTER_NEAREST);
  if(!fullsize_cloud_img)
    cv::resize(cloud_img_s, cloud_img, depth_img.size(), 0, 0, cv::INTER_NEAREST);
}
//-------------------------------------------------------------------------------------------


// Convert a normal image (each element is a normal vector) to an image
// whose element consists of alpha and beta, where:
// alpha: Angle of the normal projected on xy plane from x axis.
// beta: Angle of the normal from xy plane.
void PolarizeNormalImg(const cv::Mat &normal_img, cv::Mat &alpha_beta_img)
{
  alpha_beta_img.create(normal_img.size(), CV_32FC3);
  for(int y(0); y<normal_img.rows; ++y)
  {
    for(int x(0); x<normal_img.cols; ++x)
    {
      const cv::Vec3f &normal(normal_img.at<cv::Vec3f>(y,x));
      cv::Vec3f &alpha_beta(alpha_beta_img.at<cv::Vec3f>(y,x));
      alpha_beta[0]= std::atan2(normal[1],normal[0])/M_PI; // alpha
      alpha_beta[1]= -std::atan2(normal[2],std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]))/M_PI; // beta
      alpha_beta[2]= 0.0;
    }
  }
}
//-------------------------------------------------------------------------------------------

// Visualize a normal vector with RGB color.
template <typename t_1, typename t_2>
inline void ColorizeNormal(
    const t_1 &nx, const t_1 &ny, const t_1 &nz,
    t_2 &r, t_2 &g, t_2 &b)
{
  // Version 1
  // r= 0.5*(1.0-nx);
  // g= 0.5*(1.0-ny);
  // b= 0.5*(1.0-nz);
  // Version 2 (consider [nx,ny,nz]==[-nx,-ny,-nz])
  if(nz>=0.0)
  {
    r= 0.5*(1.0-nx);
    g= 0.5*(1.0-ny);
    b= 0.5*(1.0-nz);
  }
  else
  {
    r= 0.5*(1.0+nx);
    g= 0.5*(1.0+ny);
    b= 0.5*(1.0+nz);
  }
}
//-------------------------------------------------------------------------------------------

void ColorizeNormalImg(const cv::Mat &normal_img, cv::Mat &cnormal_img)
{
  cnormal_img.create(normal_img.size(), CV_32FC3);
  for(int y(0); y<normal_img.rows; ++y)
  {
    for(int x(0); x<normal_img.cols; ++x)
    {
      const cv::Vec3f &normal(normal_img.at<cv::Vec3f>(y,x));
      cv::Vec3f col;
      ColorizeNormal(normal[0],normal[1],normal[2], col(0),col(1),col(2));
      cnormal_img.at<cv::Vec3f>(y,x)= col;
    }
  }
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
#ifndef LIBRARY
//-------------------------------------------------------------------------------------------
#define LIBRARY
#include "ros_proj_mat.cpp"
#include "ros_capture.cpp"
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

bool is_depth(false);
double depth_scale(0.3);
cv::Mat proj_mat;

int wsize(3);
int cd2ntype(cd2ntSimple);
float resize_ratio(1.0);

float beta_min(0.0), beta_max(0.2);
double dim_image=0.5;
double dim_filtered_beta=0.7;

bool quit_at_cap_err(false);

void CVCallback(const cv::Mat &frame)
{
  if(frame.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
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

  cv::Mat normal_img, cloud_img;
  DepthImgToNormalImg(
    frame, proj_mat,
    normal_img, cloud_img, wsize, resize_ratio, /*type=*/TCD2NType(cd2ntype));  // cd2ntSimple,cd2ntRobust

  cv::Mat alpha_beta_img;
  PolarizeNormalImg(normal_img, alpha_beta_img);

  cv::Mat filtered_beta;
  cv::inRange(alpha_beta_img, cv::Scalar(-1.f,beta_min,-1.f), cv::Scalar(1.f,beta_max,1.f), filtered_beta);

  cv::Mat filtered_beta_disp,filtered_beta_col;
  filtered_beta_disp= dim_image*img_disp;
  cv::Mat filtered_beta_col_decom[3]= {128.0*filtered_beta,128.0*filtered_beta,128.0*filtered_beta+128.0*filtered_beta};
  cv::merge(filtered_beta_col_decom,3,filtered_beta_col);
  filtered_beta_disp+= dim_filtered_beta*filtered_beta_col;

  cv::Mat cnormal_img, calpha_beta_img;
  ColorizeNormalImg(normal_img, cnormal_img);
  ColorizeNormalImg(alpha_beta_img, calpha_beta_img);

  cv::imshow("input", img_disp);
  ProcMouseEvent("input", frame);

  cv::imshow("normal", normal_img);
  ProcMouseEvent("normal", normal_img);

  cv::imshow("alpha_beta", alpha_beta_img);
  ProcMouseEvent("alpha_beta", alpha_beta_img);

  cv::imshow("filtered_beta", filtered_beta_disp);
  ProcMouseEvent("filtered_beta", filtered_beta_disp);

  cv::imshow("normal_col", cnormal_img);
  ProcMouseEvent("normal_col", cnormal_img);

  cv::imshow("alpha_beta_col", calpha_beta_img);
  ProcMouseEvent("alpha_beta_col", calpha_beta_img);

  char c(cv::waitKey(1));
  if(c=='\x1b'||c=='q')  FinishLoop();
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string img_topic("/camera/aligned_depth_to_color/image_raw"), encoding;
  if(argc>1)  img_topic= argv[1];
  ros::init(argc, argv, "ros_rs_normal");
  ros::NodeHandle node("~");
  encoding= GetImageEncoding(img_topic, node, /*convert_cv=*/true);
  if(encoding=="16UC1")  is_depth= true;

  // cv::Mat proj_mat;
  std::string img_info_topic(img_topic), frame_id, ltopic("image_raw");
  img_info_topic.replace(img_info_topic.find(ltopic),ltopic.length(),"camera_info");
  GetCameraProjectionMatrix(img_info_topic, frame_id, proj_mat);

  cv::namedWindow("input",1);
  setMouseCallback("input", onMouse, "input");

  cv::namedWindow("normal",1);
  setMouseCallback("normal", onMouse, "normal");
  CreateTrackbar<int>("wsize", "normal", &wsize, 1, 15, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("cd2ntype", "normal", &cd2ntype, 0, 1, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_ratio", "normal", &resize_ratio, 0.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);

  cv::namedWindow("alpha_beta",1);
  setMouseCallback("alpha_beta", onMouse, "alpha_beta");

  cv::namedWindow("filtered_beta",1);
  setMouseCallback("filtered_beta", onMouse, "filtered_beta");
  CreateTrackbar<double>("dim_image", "filtered_beta", &dim_image, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("dim_filtered_beta", "filtered_beta", &dim_filtered_beta, 0.0, 1.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("beta_min", "filtered_beta", &beta_min, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);
  CreateTrackbar<float>("beta_max", "filtered_beta", &beta_max, -1.0f, 1.0f, 0.01f,  &TrackbarPrintOnTrack);

  cv::namedWindow("normal_col",1);
  setMouseCallback("normal_col", onMouse, "normal_col");
  cv::namedWindow("alpha_beta_col",1);
  setMouseCallback("alpha_beta_col", onMouse, "alpha_beta_col");

  StartLoop(argc, argv, img_topic, encoding, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
#endif//LIBRARY
