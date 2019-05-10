// g++ -g -Wall -O2 -o pyr_segmentation.out pyr_segmentation.cpp -lopencv_imgproc -lopencv_legacy -lopencv_core -lopencv_highgui
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>  // for cvPyrSegmentation
#include <iostream>
#include <cstdio>
#include "cap_open.h"

int main(int argc, char **argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  cv::namedWindow("camera",1);
  cv::namedWindow("segment",1);
  cv::Mat frame;
  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }
    cv::imshow("camera", frame);


    int level= 4;
    cv::Rect roi;
    roi.x= roi.y= 0;
    roi.width= frame.cols & -(1 << level);
    roi.height = frame.rows & -(1 << level);
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat src(frame, roi);
    cv::Mat dst;
    src.copyTo(dst);

    double threshold1= 255.0;
    double threshold2= 50.0;

    // CvMat src2(src), dst2(dst);
    IplImage src2(src), dst2(dst);
    CvMemStorage *storage= cvCreateMemStorage(0);
    CvSeq *comp(NULL);
    cvPyrSegmentation(&src2, &dst2, storage, &comp, level, threshold1, threshold2);
    cvReleaseMemStorage(&storage);

    cv::imshow("segment", dst);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
