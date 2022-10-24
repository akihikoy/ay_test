//-------------------------------------------------------------------------------------------
/*! \file    cv2-edge_cmp.cpp
    \brief   Comparison of edge detection methods.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.16, 2020

g++ -g -Wall -O2 -o cv2-edge_cmp.out cv2-edge_cmp.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------

// Canny
cv::Mat GetCanny(const cv::Mat &src,
  const double &threshold1=100.0,
  const double &threshold2=100.0,
  int aperture_size=3,
  int blur_size=7,
  const double &blur_std=1.5,
  bool is_depth=false,
  const double &depth_scale=0.3
)
{
  cv::Mat gray, edges;
  if(blur_size>1)
    cv::GaussianBlur(src, gray, cv::Size(blur_size,blur_size), blur_std, blur_std);
  else
    gray= src;
  if(!is_depth)
    cv::cvtColor(gray, gray, CV_BGR2GRAY);
  else
  {
    gray= depth_scale*gray;
    gray.convertTo(gray, CV_8U);
  }
  cv::Canny(gray, edges, threshold1, threshold2, aperture_size);
  return edges;
}

// Laplacian
cv::Mat GetLaplacian(const cv::Mat &src,
  int ksize=3,
  const double &scale=1.0,
  const double &delta=0.0,
  int blur_size=7,
  const double &blur_std=1.5,
  bool is_depth=false
)
{
  cv::Mat gray, edges;
  if(blur_size>1)
    cv::GaussianBlur(src, gray, cv::Size(blur_size,blur_size), blur_std, blur_std);
  else
    gray= src;
  if(!is_depth)  cv::cvtColor(gray, gray, CV_BGR2GRAY);
  cv::Laplacian(gray, edges, CV_16S, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::convertScaleAbs(edges, edges);
  return edges;
}

// Sobel
cv::Mat GetSobel(const cv::Mat &src,
  int ksize=3,
  const double &scale=1.0,
  const double &delta=0.0,
  int blur_size=7,
  const double &blur_std=1.5,
  bool is_depth=false
)
{
  cv::Mat gray, edges;
  if(blur_size>1)
    cv::GaussianBlur(src, gray, cv::Size(blur_size,blur_size), blur_std, blur_std);
  else
    gray= src;
  if(!is_depth)  cv::cvtColor(gray, gray, CV_BGR2GRAY);
  cv::Mat grad_x, grad_y;
  // Gradient X
  // if(ksize==3)
  //cv::Scharr(gray, grad_x, CV_16S, 1, 0, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(gray, grad_x, CV_16S, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::convertScaleAbs(grad_x, grad_x);
  // Gradient Y
  // if(ksize==3)
  //cv::Scharr(gray, grad_y, CV_16S, 0, 1, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(gray, grad_y, CV_16S, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::convertScaleAbs(grad_y, grad_y);
  //Merge:
  cv::Mat grads[3]= {0.0*grad_x, grad_x, grad_y};
  cv::merge(grads,3,edges);
  return edges;
}

std::ostream& operator<<(std::ostream &os, const cv::Scalar_<double> &v)
{
  os<<v(0)<<" "<<v(1)<<" "<<v(2)<<" "<<v(3);
  return os;
}


#ifndef LIBRARY
#define LIBRARY
#include "float_trackbar.cpp"

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("input",1);

  cv::namedWindow("canny",1);
  cv::namedWindow("laplacian",1);
  cv::namedWindow("sobel",1);

  double canny_threshold1=100.0;
  double canny_threshold2=200.0;
  int canny_ksize=5;
  int canny_blur_size=3;
  double canny_blur_std=1.5;
  CreateTrackbar<double>("threshold1", "canny", &canny_threshold1, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("threshold2", "canny", &canny_threshold2, 0.0, 1000.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ksize",    "canny", &canny_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<int>("blur_size","canny", &canny_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "canny", &canny_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  int laplacian_ksize=5;
  double laplacian_scale=2.0;
  double laplacian_delta=0.0;
  int laplacian_blur_size=1;
  double laplacian_blur_std=1.5;
  CreateTrackbar<int>("ksize",    "laplacian", &laplacian_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale", "laplacian", &laplacian_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta", "laplacian", &laplacian_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("blur_size","laplacian", &laplacian_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "laplacian", &laplacian_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  int sobel_ksize=3;
  double sobel_scale=6.0;
  double sobel_delta=0.0;
  int sobel_blur_size=1;
  double sobel_blur_std=1.5;
  CreateTrackbar<int>("ksize",    "sobel", &sobel_ksize, 3, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("scale", "sobel", &sobel_scale, 0.0, 50.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("delta", "sobel", &sobel_delta, -255.0, 255.0, 0.01, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("blur_size","sobel", &sobel_blur_size, 1, 25, 2,  &TrackbarPrintOnTrack);
  CreateTrackbar<double>("blur_std", "sobel", &sobel_blur_std, 0.01, 10.0, 0.01, &TrackbarPrintOnTrack);

  cv::Mat frame;
  cv::Mat canny, laplacian, sobel;
  std::vector<cv::Mat> mask_points(1);
  bool quit_at_cap_err(false), disp_sum(false), using_mask(false);
  double sumscale(1.0e-6);
  /*create mask (use ./mouse_poyly.out to get the points)*/{
    cv::Mat points(6,2,CV_32S);
    points= (cv::Mat_<int>(6,2)<<
          0, 123,
          0, 401,
          300, 300,
          639, 393,
          628, 114,
          300, 228);
    mask_points[0]= points;
  }
  for(int i(0);;++i)
  {
    if(!cap.Read(frame))
    {
      if(!quit_at_cap_err && cap.WaitReopen()) continue;
      else break;
    }
    if(using_mask)
    {
      cv::Mat mask(frame.size(), CV_8U, cv::Scalar(0)), masked;
      cv::fillPoly(mask, mask_points, cv::Scalar(255));
      frame.copyTo(masked, mask);
      frame= masked;
    }

    canny= GetCanny(frame,
      canny_threshold1, canny_threshold2, canny_ksize,
      canny_blur_size, canny_blur_std);
    laplacian= GetLaplacian(frame,
      laplacian_ksize, laplacian_scale, laplacian_delta,
      laplacian_blur_size, laplacian_blur_std);
    sobel= GetSobel(frame,
      sobel_ksize, sobel_scale, sobel_delta,
      sobel_blur_size, sobel_blur_std);

    cv::imshow("input", frame);
    cv::imshow("canny", canny);
    cv::imshow("laplacian", laplacian);
    cv::imshow("sobel", sobel);

    if(disp_sum)
      std::cout<<"canny, laplacian, sobel: "<<cv::sum(canny)*sumscale<<", "<<cv::sum(laplacian)*sumscale<<", "<<cv::sum(sobel)*sumscale<<std::endl;
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  disp_sum=!disp_sum;
    if(c=='m')  using_mask=!using_mask;
  }

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
