//-------------------------------------------------------------------------------------------
/*! \file    cv2-laplacian2.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.08, 2017

g++ -g -Wall -O2 -o cv2-laplacian2.out cv2-laplacian2.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

using namespace cv;

const char* window_name = "Laplace+Threshold";
int lthreshold(10);
Mat src;

void LaplacianThreshold(int,void*)
{
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  Mat src_gray, dst;

  /// Remove noise by blurring with a Gaussian filter
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Apply Laplace function
  Mat abs_dst;

  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );

  cv::threshold(abs_dst, abs_dst, lthreshold, 255.0, cv::THRESH_BINARY);

  imshow( window_name, abs_dst );
  // imshow( window_name, dst );
}

int main( int argc, char** argv )
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;


  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  createTrackbar( "Threshold:", window_name, &lthreshold, 1000, LaplacianThreshold);

  namedWindow( "cam", CV_WINDOW_AUTOSIZE );

  for(;;)
  {
    cap >> src;

    LaplacianThreshold(0,0);

    imshow( "cam", src );

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }

  return 0;
}
