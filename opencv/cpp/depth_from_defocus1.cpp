//-------------------------------------------------------------------------------------------
/*! \file    depth_from_defocus1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.08, 2017

g++ -g -Wall -O2 -o depth_from_defocus1.out depth_from_defocus1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cap_open.h"

int main(int argc, char **argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::namedWindow("test",1);
  // int dthreshold(0);
  // cv::createTrackbar( "Threshold:", "test", &dthreshold, 1000, NULL);
  for(;;)
  {
    cv::Mat frame, frame_gray, frame_edge, frame_blur1, frame_blur2;
    cap >> frame;
    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);

    float std= 1.0;
    float std1= std;
    float std2= 1.5*std;

    cv::Canny(frame_gray, frame_edge, 0, 30, 3);

    int w1=(2*std::ceil(2*std1))+1, w2=(2*std::ceil(2*std2))+1;
    cv::GaussianBlur(frame_gray, frame_blur1, cv::Size(w1,w1), std1, std1);
    cv::GaussianBlur(frame_gray, frame_blur2, cv::Size(w2,w2), std2, std2);

    cv::Laplacian(frame_blur1, frame_blur1, CV_16S, /*kernel_size*/3, /*scale*/1, /*delta*/0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(frame_blur1,frame_blur1);
    cv::Laplacian(frame_blur2, frame_blur2, CV_16S, /*kernel_size*/3, /*scale*/1, /*delta*/0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(frame_blur2,frame_blur2);


    cv::Mat sparse_dmap(frame_gray.size(), CV_32FC1);
    cv::Mat sparse_dmap2(frame_gray.size(), CV_32FC1);
    cv::MatIterator_<float>
      itr_sdm= sparse_dmap.begin<float>(),
      itr_sdm_end = sparse_dmap.end<float>();
    cv::MatIterator_<short>
      itr_fbl1= frame_blur1.begin<short>(),
      itr_fbl2= frame_blur2.begin<short>();
    cv::MatIterator_<uchar>
      itr_fedg= frame_edge.begin<uchar>();
    for(; itr_sdm!=itr_sdm_end; ++itr_sdm,++itr_fbl1,++itr_fbl2,++itr_fedg)
    {
      *itr_sdm= 0.0;
      if(*itr_fedg>0 && *itr_fbl2!=0)
      {
        float gratio= float(*itr_fbl1)/float(*itr_fbl2);
        if(gratio>1.02 && gratio<=(std2/std1))
          *itr_sdm= sqrt((gratio*gratio*std1*std1-std2*std2)/(1.0-gratio*gratio));
// std::cerr<<" | "<<int(*itr_fbl1)<<" "<<int(*itr_fbl2)<<" "<<gratio<<" "<<*itr_sdm;
      }
    }

    cv::bilateralFilter(sparse_dmap, sparse_dmap2, /*d*/5, /*sigmaColor*/4.0, /*sigmaSpace*/5.0);

    // cv::erode(sparse_dmap2,sparse_dmap2,cv::Mat(),cv::Point(-1,-1), 5);
    // cv::dilate(sparse_dmap2,sparse_dmap2,cv::Mat(),cv::Point(-1,-1), 5);

    // cv::threshold(sparse_dmap2, sparse_dmap2, dthreshold/500.0, 255.0, cv::THRESH_BINARY);

    // cv::normalize(sparse_dmap2, sparse_dmap2, 0, 255, CV_MINMAX, CV_8U);

    imshow("cam", frame_gray);
    imshow("test", sparse_dmap2);
    char c(cv::waitKey(1));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
