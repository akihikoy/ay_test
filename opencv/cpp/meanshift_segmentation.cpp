// Meanshift sample
// src: https://code.ros.org/trac/opencv/browser/trunk/opencv/samples/cpp/meanshift_segmentation.cpp?rev=3944
// g++ -g -Wall -O2 -o meanshift_segmentation.out meanshift_segmentation.cpp -lopencv_imgproc -lopencv_legacy -lopencv_core -lopencv_highgui
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include "cap_open.h"

using namespace cv;
using namespace std;

void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}

string winName = "meanshift";
int spatialRad, colorRad, maxPyrLevel;
Mat img, res;

void meanShiftSegmentation( int, void* )
{
    cout << "spatialRad=" << spatialRad << "; "
         << "colorRad=" << colorRad << "; "
         << "maxPyrLevel=" << maxPyrLevel << endl;
    TermCriteria termcrit( TermCriteria::MAX_ITER+TermCriteria::EPS,5,50.0);
    pyrMeanShiftFiltering( img, res, spatialRad, colorRad, maxPyrLevel, termcrit );
    floodFillPostprocess( res, Scalar::all(2) );
    imshow( winName, res );
}

int main(int argc, char **argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);
  cv::namedWindow(winName,1);

  spatialRad = 10;
  colorRad = 10;
  maxPyrLevel = 1;
  createTrackbar( "spatialRad", winName, &spatialRad, 80, meanShiftSegmentation );
  createTrackbar( "colorRad", winName, &colorRad, 60, meanShiftSegmentation );
  createTrackbar( "maxPyrLevel", winName, &maxPyrLevel, 5, meanShiftSegmentation );

  for(;;)
  {
    // cap >> img;
    if(!cap.Read(img))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    cv::imshow("camera", img);


    meanShiftSegmentation(0, 0);



    int c(cv::waitKey(10));
    // int c(cv::waitKey(200));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
