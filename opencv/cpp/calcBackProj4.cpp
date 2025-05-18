// g++ -g -I -Wall calcBackProj4.cpp -o calcBackProj4.out -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui -I/usr/include/opencv4

// based on opencv/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo2.cpp

// NOTE: HSV version, but no big change from HS version in practice.
// Use HS version: calcBackProj3.cpp
// NOTE-2: However when you want to back-project black colors, using HSV is better.

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "cap_open.h"

using namespace cv;
using namespace std;

/// Global Variables
Mat src; Mat hsv;
Mat mask;

int lo = 20; int up = 20;
const char* window_image = "Source image";

// Histogram parameters
MatND hist;
float h_range[] = { 0, 179 };
float s_range[] = { 0, 255 };
float v_range[] = { 0, 255 };
const float* ranges[] = { h_range, s_range, v_range };
int channels[] = { 0, 1, 2 };
int h_bins = 30; int s_bins = 32; int v_bins = 30;
int histSize[] = { h_bins, s_bins, v_bins };

/// Function Headers
void Backproj( );
void pickPoint (int event, int x, int y, int, void* );

/**
 * @function main
 */
int main( int argc, char** argv )
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  namedWindow( window_image, WINDOW_AUTOSIZE );

  /// Set Trackbars for floodfill thresholds
  createTrackbar( "Low thresh", window_image, &lo, 255, 0 );
  createTrackbar( "High thresh", window_image, &up, 255, 0 );
  /// Set a Mouse Callback
  setMouseCallback( window_image, pickPoint, 0 );

  for(int i(0);;++i)
  {
    cv::Mat frame;
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    src= frame;

    /// Transform it to HSV
    cvtColor( src, hsv, COLOR_BGR2HSV );

    imshow( window_image, src );

    if(i==0)
    {
      /// Get the Histogram and normalize it
      calcHist( &hsv, 1, channels, mask, hist, 3, histSize, ranges, true, false );
      // normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
      // threshold( hist, hist, 255, 255, THRESH_TRUNC);
      // normalize( hist[0], hist[0], 0, 255, NORM_MINMAX, -1, Mat() );
      // normalize( hist[1], hist[1], 0, 255, NORM_MINMAX, -1, Mat() );
      // normalize( hist[2], hist[2], 0, 255, NORM_MINMAX, -1, Mat() );
      for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s) for(int v(0);v<v_bins;++v)
        if(hist.at<float>(h,s,v)>255)  hist.at<float>(h,s,v)= 255;
    }

    Backproj( );

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}

/**
 * @function pickPoint
 */
void pickPoint (int event, int x, int y, int, void* )
{
  if( event != cv::EVENT_LBUTTONDOWN )
    { return; }

  // Fill and get the mask
  Point seed = Point( x, y );

  int newMaskVal = 255;
  Scalar newVal = Scalar( 120, 120, 120 );

  int connectivity = 8;
  int flags = connectivity + (newMaskVal << 8 ) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;

  Mat mask2 = Mat::zeros( src.rows + 2, src.cols + 2, CV_8UC1 );
  floodFill( src, mask2, seed, newVal, 0, Scalar( lo, lo, lo ), Scalar( up, up, up), flags );
  mask = mask2( Range( 1, mask2.rows - 1 ), Range( 1, mask2.cols - 1 ) );

  imshow( "Mask", mask );


  /// Get the Histogram and normalize it
  calcHist( &hsv, 1, channels, mask, hist, 3, histSize, ranges, true, false );
// for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s) for(int v(0);v<v_bins;++v)
// std::cerr<<" "<<hist.at<float>(h,s,v);
// std::cerr<<std::endl;
// std::cerr<<hist.size[0]<<" "<<hist.size[1]<<" "<<hist.size[2]<<std::endl;
  // normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );
  // threshold( hist, hist, 255, 255, THRESH_TRUNC);
  for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s) for(int v(0);v<v_bins;++v)
    if(hist.at<float>(h,s,v)>255)  hist.at<float>(h,s,v)= 255;
// std::cerr<<hist.size[0]<<" "<<hist.size[1]<<" "<<hist.size[2]<<std::endl;
// for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s) for(int v(0);v<v_bins;++v)
// std::cerr<<" "<<hist.at<float>(h,s,v);
// std::cerr<<std::endl;
  // normalize( hist[0], hist[0], 0, 255, NORM_MINMAX, -1, Mat() );
  // normalize( hist[1], hist[1], 0, 255, NORM_MINMAX, -1, Mat() );
  // normalize( hist[2], hist[2], 0, 255, NORM_MINMAX, -1, Mat() );

  Backproj( );
}

/**
 * @function Backproj
 */
void Backproj( )
{
  /// Get Backprojection
  MatND backproj;
  calcBackProject( &hsv, 1, channels, hist, backproj, ranges, 5, true );

  /// Draw the backproj
  imshow( "BackProj (HSV)", backproj );

}
