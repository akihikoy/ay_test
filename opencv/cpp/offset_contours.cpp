//-------------------------------------------------------------------------------------------
/*! \file    offset_contours.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.19, 2025

g++ -g -Wall -O2 -o offset_contours.out offset_contours.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

$ ./offset_contours.out sample/binary1.png
$ ./offset_contours.out sample/opencv-logo.png
$ ./offset_contours.out sample/water_coins.jpg
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define LIBRARY
#include "float_trackbar.cpp"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

// Calculate normal for OffsetContours.
cv::Point2f CalculateNormal(const cv::Point &p1, const cv::Point &p2)
{
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  return cv::Point2f(-dy, dx);
}

// Offset contours_in with offset (positive values for outside offset).
// Note that this function implements a simple algorithm, just offsetting
// each vertex to a normal direction computed from neighbor points.
// It does not consider carefully the corner points.
// It may provide some output for negative offset, but the output may be unexpected one.
// The result is stored into contours_out.
void OffsetContours(
    const std::vector<std::vector<cv::Point> >& contours_in,
    std::vector<std::vector<cv::Point> >& contours_out, float offset)
{
  contours_out.clear();

  for(const auto& contour : contours_in)
  {
    std::vector<cv::Point> offsetContour;
    int n= contour.size();

    for(int i(0); i < n; ++i)
    {
      // Computing a normal from surrounding points.
      cv::Point p1= contour[i];
      cv::Point p2= contour[(i + 1) % n];
      cv::Point p0= contour[(i - 1 + n) % n];

      cv::Point2f normal1= CalculateNormal(p0, p1);
      cv::Point2f normal2= CalculateNormal(p1, p2);
      cv::Point2f normal = (normal1 + normal2);
      float length= std::sqrt(normal.x * normal.x + normal.y * normal.y);

      normal*= offset / length;

      // Offset a point.
      cv::Point newPoint(static_cast<int>(p1.x + normal.x), static_cast<int>(p1.y + normal.y));
      offsetContour.push_back(newPoint);
    }
    contours_out.push_back(offsetContour);
  }
}
//------------------------------------------------------------------------------------------

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
  std::string filename("sample/water_coins.jpg");
  if(argc>1)  filename= argv[1];
  cv::Mat img_in= cv::imread(filename, cv::IMREAD_COLOR);

  float resize_factor(1.0), offset(0.0);
  bool inv_img(false);
  int n_erode(2), n_dilate(2);
  cv::namedWindow("Input",1);
  CreateTrackbar<bool>("Invert", "Input", &inv_img, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_factor", "Input", &resize_factor, 0.1, 5.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_dilate", "Input", &n_dilate, 0, 15, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("n_erode", "Input", &n_erode, 0, 15, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("offset", "Input", &offset, -10.0, 10.0, 0.1, &TrackbarPrintOnTrack);

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
    std::vector<std::vector<cv::Point> > contours, contours_offset;
    cv::findContours(img_binary, contours, /*cv::RETR_EXTERNAL*/cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    OffsetContours(contours, contours_offset, offset);

    /// Visualize results
    cv::Mat img_binary_col;
    cv::Mat img_binary_col_decom[3]= {img_binary,0.0*img_binary,0.0*img_binary};
    cv::merge(img_binary_col_decom,3,img_binary_col);
    img*= 0.5;
    img+= img_binary_col;
    for(int ic(0),ic_end(contours.size()); ic<ic_end; ++ic)
      cv::drawContours(img, contours, ic, cv::Scalar(255,0,0), /*thickness=*/1, /*linetype=*/8);
    for(int ic(0),ic_end(contours_offset.size()); ic<ic_end; ++ic)
      cv::drawContours(img, contours_offset, ic, cv::Scalar(255,0,255), /*thickness=*/1, /*linetype=*/8);
    for(int ic(0),ic_end(contours_offset.size()); ic<ic_end; ++ic)
      for(int ip(0),ip_end(contours_offset[ic].size()); ip<ip_end; ++ip)
        cv::circle(img, contours_offset[ic][ip], 2, cv::Scalar(255,0,255));
    cv::imshow("Input", img);

    char c(cv::waitKey(500));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
