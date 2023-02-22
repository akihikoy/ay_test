//-------------------------------------------------------------------------------------------
/*! \file    ros_rs_sumpoly2.cpp
    \brief   Calculate sum of depth/rgb/normal/point images within a polygon(2).
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.22, 2023

$ g++ -O2 -g -W -Wall -o ros_rs_sumpoly2.out ros_rs_sumpoly2.cpp -I/opt/ros/$ROS_DISTR/include -pthread -llog4cxx -lpthread -L/opt/ros/$ROS_DISTR/lib -rdynamic -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lcv_bridge -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_videoio -Wl,-rpath,/opt/ros/$ROS_DISTR/lib
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

struct TCalcOverPolygonOperator
{
  virtual void operator()(int x, int y) = 0;
};

// Apply an operation over a polygon.
void CalcOverPolygon(
  const cv::Size &img_size, const std::vector<cv::Point> &polygon,
  TCalcOverPolygonOperator &op)
{
  // Copy a patch of the bounding box of the polygon from the depth image.
  cv::Rect bb= cv::boundingRect(polygon);
  if(bb.width*bb.height==0)  return;
  if(bb.x<0 || bb.y<0 || bb.x+bb.width>img_size.width || bb.y+bb.height>img_size.height)  return;

  // Mask by the polygon.
  std::vector<std::vector<cv::Point> > polygon_pts(1);
  polygon_pts[0]= polygon;
  for(std::vector<cv::Point>::iterator itr(polygon_pts[0].begin()),itr_end(polygon_pts[0].end()); itr!=itr_end; ++itr)
  {
    itr->x-= bb.x;
    itr->y-= bb.y;
  }

  cv::Mat polygon_mask_img= cv::Mat::zeros(bb.size(), CV_8UC1);
  cv::fillPoly(polygon_mask_img, polygon_pts, 1);

  for(int y(0);y<bb.height;++y)
    for(int x(0);x<bb.width;++x)
      if(polygon_mask_img.at<unsigned char>(y,x))
        op(x+bb.x, y+bb.y);
}
//-------------------------------------------------------------------------------------------


// Compute a sum of depth, average and std-dev of normal inside a polygon.
void SumDepthAverageNormalWithinPolygon(
  const cv::Mat &img_depth,
  const cv::Mat &normal_img,
  const std::vector<cv::Point> &polygon,
  unsigned int depth_range[2],
  float &sum_depth,
  cv::Vec3f &avr_normal,
  int &num_pixels)
{
  struct t_operator : TCalcOverPolygonOperator
  {
    const cv::Mat &img_depth_;
    const cv::Mat &normal_img_;
    unsigned int depth_range_[2];
    float sum_depth_;
    int counter_depth_;
    cv::Vec3f sum_normal_;
    int counter_normal_;
    t_operator(
        const cv::Mat &img_depth,
        const cv::Mat &normal_img,
        unsigned int depth_range[2])
        : img_depth_(img_depth), normal_img_(normal_img),
          sum_depth_(0.0), counter_depth_(0), sum_normal_(0,0,0), counter_normal_(0)
      {
        depth_range_[0]= depth_range[0];
        depth_range_[1]= depth_range[1];
      }
    void operator()(int x, int y)
      {
        const unsigned short &depth(img_depth_.at<unsigned short>(y,x));
        ++counter_depth_;
        if(depth_range_[0]<=depth && depth<=depth_range_[1])
          sum_depth_+= depth_range_[1]-depth;

        const cv::Vec3f &normal(normal_img_.at<cv::Vec3f>(y,x));
        float norm= cv::norm(normal);
        if(std::fabs(norm-1.0f)<0.1)
        {
          ++counter_normal_;
          sum_normal_+= normal;
        }
      }
  };
  t_operator op(img_depth,normal_img,depth_range);
  CalcOverPolygon(img_depth.size(), polygon, op);

  sum_depth= op.sum_depth_;
  avr_normal= op.sum_normal_/op.counter_normal_;
  avr_normal/= cv::norm(avr_normal);
  num_pixels= op.counter_depth_;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
#ifndef LIBRARY
//-------------------------------------------------------------------------------------------
#define LIBRARY
#include "ros_proj_mat.cpp"
#include "ros_rs_normal.cpp"
#include "ros_capture2.cpp"
#include "float_trackbar.cpp"
#include "cv2-print_elem.cpp"
#include "ros_rs_sumpoly.cpp"
//-------------------------------------------------------------------------------------------


std::vector<std::vector<cv::Point> >  polygon(1);

bool mouse_event_detected(false);
int x_mouse(0), y_mouse(0), mouse_event(0), mouse_flags(0);

void OnMouse(int event, int x, int y, int flags, void *)
{
  if(event==cv::EVENT_LBUTTONDOWN || event==cv::EVENT_LBUTTONUP
    || event==cv::EVENT_RBUTTONDOWN || event==cv::EVENT_RBUTTONUP)
  {
    mouse_event_detected= true;
    x_mouse= x; y_mouse= y;
    mouse_event= event;
    mouse_flags= flags;
  }
}
//-------------------------------------------------------------------------------------------

void ProcMouseEvent(
    const cv::Mat &frame_rgb, const cv::Mat &frame_depth,
    const cv::Mat &normal_img, const cv::Mat &cloud_img)
{
  if(!mouse_event_detected)  return;

  if(mouse_event==cv::EVENT_LBUTTONDOWN/* && mouse_flags==0*/)
  {
    std::cout<<"Clicked:["<<x_mouse<<","<<y_mouse<<"]:"
        <<" RGB:"<<GetPixelVal(frame_rgb,x_mouse,y_mouse)
        <<" Depth:"<<GetPixelVal(frame_depth,x_mouse,y_mouse)
        <<" Normal:"<<GetPixelVal(normal_img,x_mouse,y_mouse)
        <<" Pt3d:"<<GetPixelVal(cloud_img,x_mouse,y_mouse)
        <<std::endl;
  }
  if(mouse_event==cv::EVENT_LBUTTONDOWN && mouse_flags&cv::EVENT_FLAG_SHIFTKEY)
  {
    polygon[0].push_back(cv::Point(x_mouse,y_mouse));
  }
  if(mouse_event==cv::EVENT_RBUTTONDOWN && mouse_flags&cv::EVENT_FLAG_SHIFTKEY)
  {
    std::cout<<"Flushing polygon:";
    for(size_t i(0); i<polygon[0].size(); ++i)
      std::cout<<" "<<polygon[0][i];
    std::cout<<std::endl;
    polygon[0].clear();
  }
  mouse_event_detected= false;
}
//-------------------------------------------------------------------------------------------

double depth_scale(0.3);
bool quit_at_cap_err(false);
cv::Mat proj_mat;

int wsize(5);
int cd2ntype(cd2ntSimple);
float resize_ratio(0.5);

int depth_min(0), depth_max(600);

void CVCallback(const cv::Mat &frame_depth, const cv::Mat &frame_rgb)
{
  if(frame_depth.empty() || frame_rgb.empty())
  {
    if(quit_at_cap_err)  FinishLoop();
    return;
  }

  cv::Mat normal_img, cloud_img;
  DepthImgToNormalImg(
    frame_depth, proj_mat,
    normal_img, cloud_img, wsize, resize_ratio, /*type=*/TCD2NType(cd2ntype));  // cd2ntSimple,cd2ntRobust

  if(mouse_event_detected)
  {
    ProcMouseEvent(frame_rgb, frame_depth, normal_img, cloud_img);

    std::cout<<"Image stats in the polygon:"<<std::endl;

    unsigned int depth_range[2]= {(unsigned int)depth_min, (unsigned int)depth_max};
    float sum_depth(0.0);
    int num_pixels(0);
    SumDepthWithinPolygon(frame_depth, polygon[0], depth_range, sum_depth, num_pixels);
    std::cout<<"  sum_depth:"<<sum_depth<<" num_pixels:"<<num_pixels<<" avr_depth:"<<sum_depth/(float)num_pixels<<std::endl;
    cv::Vec3f avr_normal;
    AverageNormalWithinPolygon(normal_img, polygon[0], avr_normal, num_pixels);
    std::cout<<"  avr_normal:"<<avr_normal<<" num_pixels:"<<num_pixels<<std::endl;
    SumDepthAverageNormalWithinPolygon(frame_depth, normal_img, polygon[0], depth_range, sum_depth, avr_normal, num_pixels);
    std::cout<<"  sum_depth(new):"<<sum_depth<<" num_pixels:"<<num_pixels<<" avr_depth:"<<sum_depth/(float)num_pixels<<std::endl;
    std::cout<<"  avr_normal(new):"<<avr_normal<<std::endl;
  }

  cv::polylines(frame_rgb, polygon, /*isClosed=*/true, CV_RGB(255,0,255), 2);
  cv::polylines(frame_depth, polygon, /*isClosed=*/true, 800, 2);
  cv::polylines(normal_img, polygon, /*isClosed=*/true, cv::Vec3f(1,0,1), 2);
  cv::polylines(cloud_img, polygon, /*isClosed=*/true, cv::Vec3f(1,0,1), 2);

  cv::Mat img_depth_disp(frame_depth*depth_scale);
  img_depth_disp.convertTo(img_depth_disp, CV_8U);
  cv::cvtColor(img_depth_disp, img_depth_disp, CV_GRAY2BGR);
  cv::imshow("input_depth", img_depth_disp);

  cv::imshow("input_rgb", frame_rgb);

  cv::imshow("normal", normal_img);
  cv::imshow("points3d", cloud_img);

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
  ros::init(argc, argv, "ros_rs_sumpoly2");
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
  cv::setMouseCallback("input_depth", OnMouse);
  cv::namedWindow("input_rgb",1);
  cv::setMouseCallback("input_rgb", OnMouse);
  cv::namedWindow("normal",1);
  cv::setMouseCallback("normal", OnMouse);
  cv::namedWindow("points3d",1);
  cv::setMouseCallback("points3d", OnMouse);

  CreateTrackbar<int>("depth_min", "input_depth", &depth_min, 0, 2000, 1,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("depth_max", "input_depth", &depth_max, 0, 2000, 1,  &TrackbarPrintOnTrack);

  StartLoop(argc, argv, img1_topic, img2_topic, encoding1, encoding2, CVCallback, /*node_name=*/"");
  return 0;
}
//-------------------------------------------------------------------------------------------
#endif//LIBRARY
