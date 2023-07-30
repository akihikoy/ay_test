/*

g++ -g -Wall -O2 -o thinning_test.out thinning_test.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

./thinning_test.out ../sample/binary1.png
./thinning_test.out ../sample/opencv-logo.png
./thinning_test.out ../sample/water_coins.jpg

*/

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "thinning.hpp"

#include "thinning.cpp"

#include <cstdio>
#include <sys/time.h>  // gettimeofday

#define LIBRARY
#include "../float_trackbar.cpp"

using namespace std;
using namespace cv;


inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
}

int main(int argc, char **argv)
{
  Mat img_in = imread(argv[1], IMREAD_COLOR);

  float resize_factor(0.5);
  bool inv_img(false);
  cv::namedWindow("Input",1);
  CreateTrackbar<bool>("Invert", "Input", &inv_img, &TrackbarPrintOnTrack);
  CreateTrackbar<float>("resize_factor", "Input", &resize_factor, 0.1, 1.0, 0.1, &TrackbarPrintOnTrack);

  while(true)
  {
    Mat img;
    if(resize_factor!=1.0)
      resize(img_in, img, Size(), resize_factor, resize_factor, INTER_LINEAR);
    else
      img= img_in;

    /// Threshold the input image
    Mat img_grayscale, img_binary;
    cvtColor(img, img_grayscale,COLOR_BGR2GRAY);
    if(inv_img)
      threshold(img_grayscale, img_binary, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);
    else
      threshold(img_grayscale, img_binary, 0, 255, THRESH_OTSU | THRESH_BINARY);

    /// Apply thinning to get a skeleton
    Mat img_thinning_ZS, img_thinning_GH;
    double t0= GetCurrentTime();
    ximgproc::thinning(img_binary, img_thinning_ZS, ximgproc::THINNING_ZHANGSUEN);
    double t1= GetCurrentTime();
    ximgproc::thinning(img_binary, img_thinning_GH, ximgproc::THINNING_GUOHALL);
    double t2= GetCurrentTime();

    std::cout<<"Computation time:"<<endl
      <<"  THINNING_ZHANGSUEN: "<<t1-t0<<" sec"<<endl
      <<"  THINNING_GUOHALL: "<<t2-t1<<" sec"<<endl;

    /// Visualize results
    Mat result_ZS(img.rows, img.cols, CV_8UC3), result_GH(img.rows, img.cols, CV_8UC3);
    Mat in[] = { img_thinning_ZS, img_thinning_ZS, img_thinning_ZS };
    Mat in2[] = { img_thinning_GH, img_thinning_GH, img_thinning_GH };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels( in, 3, &result_ZS, 1, from_to, 3 );
    mixChannels( in2, 3, &result_GH, 1, from_to, 3 );
    result_ZS= 0.5*img + result_ZS;
    result_GH= 0.5*img + result_GH;
    imshow("Input", img_in);
    imshow("Thinning ZHANGSUEN", result_ZS);
    imshow("Thinning GUOHALL", result_GH);

    char c(cv::waitKey(500));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
