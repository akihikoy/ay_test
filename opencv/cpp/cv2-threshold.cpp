//-------------------------------------------------------------------------------------------
/*! \file    cv2-threshold.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.06, 2016

g++ -g -Wall -O2 -o cv2-threshold.out cv2-threshold.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "cap_open.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

using namespace cv;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 1;

Mat src, src_gray, dst, modified;
const char* window_name = "Threshold Demo";

const char* trackbar_type = "THRESH_: \n 0: BINARY \n 1: BINARY_INV \n 2: TRUNC \n 3: TOZERO \n 4: TOZERO_INV";
const char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo( int, void* );

/**
 * @function main
 */
int main( int argc, char** argv )
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cap >> src;
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window to display results
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  namedWindow( "camera", CV_WINDOW_AUTOSIZE );
  namedWindow( "emphasized", CV_WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, NULL );

  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, NULL );

  /// Wait until user finishes program
  while(true)
  {
    cap >> src;
    cvtColor( src, src_gray, CV_BGR2GRAY );

    /* 0: Binary
      1: Binary Inverted
      2: Threshold Truncated
      3: Threshold to Zero
      4: Threshold to Zero Inverted
    */
    threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
    modified= 0.5*src;
    cv::Mat maskbgr[3]= {dst*128.0,dst*128.0,dst*0.0}, maskcol;
    // modified+= Scalar(128.0,128.0,0.0)*dst;
    cv::merge(maskbgr,3,maskcol);
    modified+= maskcol;
    imshow( window_name, dst );
    imshow( "camera", src );
    imshow( "emphasized", modified );

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

}

