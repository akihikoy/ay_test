//-------------------------------------------------------------------------------------------
/*! \file    refine_contours1.cpp
    \brief   Sample of refining contours.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.24, 2023

g++ -g -Wall -O2 -o refine_contours1.out refine_contours1.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

$ ./refine_contours1.out sample/binary1.png
$ ./refine_contours1.out sample/opencv-logo.png
$ ./refine_contours1.out sample/water_coins.jpg
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------

// Refine a given contour (list of points).  If a line segment is longer than min_dist, it is adjusted to be shorter than min_dist.
void RefineContour(const std::vector<cv::Point> &contour_in, std::vector<cv::Point> &contour_out, const float &min_dist)
{
  contour_out.clear();
  if(contour_in.size()==0)  return;
  contour_out.push_back(contour_in[0]);
  for(int ip(1),ip_end(contour_in.size()); ip<=ip_end; ++ip)
  {
    bool is_last= (ip==ip_end);  // For considering the last-first segment.
    cv::Point dir= !is_last ? (contour_in[ip]-contour_in[ip-1]) : (contour_in[0]-contour_in[ip-1]);
    float d= cv::norm(dir);
    for(int iadd(0),iadd_num(d/min_dist); iadd<iadd_num; ++iadd)
      contour_out.push_back(contour_in[ip-1] + float(iadd+1)*dir/float(iadd_num+1));
    if(!is_last)  contour_out.push_back(contour_in[ip]);
  }
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string filename("sample/water_coins.jpg");
  if(argc>1)  filename= argv[1];
  cv::Mat img_in= cv::imread(filename, cv::IMREAD_COLOR);

  float resize_factor(1.0);
  bool inv_img(false);
  int n_erode(2), n_dilate(2);
  float min_dist(10);
  cv::namedWindow("Input",1);
  CreateTrackbar<bool>("Invert", "Input", &inv_img, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_factor", "Input", &resize_factor, 0.1, 5.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_dilate", "Input", &n_dilate, 0, 15, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_erode", "Input", &n_erode, 0, 15, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("min_dist", "Input", &min_dist, 0.01, 20.0, 0.01, &TrackbarPrintOnTrack);

  while(true)
  {
    cv::Mat img;
    if(resize_factor!=1.0)
      cv::resize(img_in, img, cv::Size(), resize_factor, resize_factor, cv::INTER_LINEAR);
    else
      img_in.copyTo(img);

    /// Threshold the input image
    cv::Mat img_grayscale, img_binary;
    cv::cvtColor(img, img_grayscale, cv::COLOR_BGR2GRAY);
    if(inv_img)
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
    else
      cv::threshold(img_grayscale, img_binary, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    cv::erode(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1), n_erode);
    cv::dilate(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1), n_dilate);

    /// Find contours.
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(img_binary, contours, /*cv::RETR_EXTERNAL*/cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    /// Refine contours.
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
    {
      std::vector<cv::Point> contour_out;
      RefineContour(contours[ic], contour_out, min_dist);
      contours[ic]= contour_out;
    }

    /// Visualize results
    cv::Mat img_binary_col;
    cv::Mat img_binary_col_decom[3]= {img_binary,0.0*img_binary,0.0*img_binary};
    cv::merge(img_binary_col_decom,3,img_binary_col);
    img*= 0.5;
    img+= img_binary_col;
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      cv::drawContours(img, contours, ic, cv::Scalar(255,0,255), /*thickness=*/1, /*linetype=*/8);
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      for(int ip(0),ip_end(contours[ic].size()); ip<ip_end; ++ip)
        cv::circle(img, contours[ic][ip], 2, cv::Scalar(255,0,255));
    cv::imshow("Input", img);

    char c(cv::waitKey(500));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
