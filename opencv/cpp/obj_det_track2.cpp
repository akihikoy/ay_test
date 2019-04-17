//-------------------------------------------------------------------------------------------
/*! \file    obj_det_track2.cpp
    \brief   Object detection and tracking
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.14, 2017

NOTE: Found that the approach of obj_det_track1.cpp is better.

g++ -I -Wall obj_det_track2.cpp -o obj_det_track2.out -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

int main(int argc, char**argv)
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  cv::BackgroundSubtractorMOG2 bkg_sbtr(/*history=*/30, /*varThreshold=*/10.0, /*detectShadows=*/true);

  cv::namedWindow("BkgSbtr",1);
  int history(30);
  cv::createTrackbar( "History:", "BkgSbtr", &history, 100, NULL);

  // Histogram
  cv::SparseMat hist, hist_tmp;
  float h_range[] = { 0, 179 };
  float s_range[] = { 0, 255 };
  int h_bins = 20; int s_bins = 10;
  #define N_CHANNELS 2
  // #define N_CHANNELS 1
  #if N_CHANNELS==2
  const float* ranges[] = { h_range, s_range };
  int channels[] = { 0, 1 }, n_channels= 2;
  int histSize[] = { h_bins, s_bins };
  #elif N_CHANNELS==1
  const float* ranges[] = { h_range };
  int channels[] = { 0 }, n_channels= 1;
  int histSize[] = { h_bins };
  #endif

  cv::Mat frame, frame_hsv, mask, frame_masked;
  bool running(true), detecting(true);
  for(int i(0);;++i)
  {
    if(running)
    {
      cap >> frame; // get a new frame from camera
      cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

      // if(detecting)
      // {
      bkg_sbtr(frame, mask, 1./float(history));
      cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 3);
      // cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), 2);
      // cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), 3);

      // if(i>1)
      // {
        // std::cerr<<"---"<<i;
        // for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
        // std::cerr<<" "<<hist.at<float>(h,s);
        // std::cerr<<std::endl;
      // }

      if(detecting)
      {
        // Get the Histogram and normalize it
        cv::calcHist(&frame_hsv, 1, channels, mask, hist_tmp, n_channels, histSize, ranges, true, false);
        // Add to the object model
        if(i==0)  hist_tmp.copyTo(hist);
        else
        {
          for(cv::SparseMatIterator_<float>
                itr(hist_tmp.begin<float>()),
                itr_end(hist_tmp.end<float>()); itr!=itr_end; ++itr)
            hist.ref<float>(itr.node()->idx)+= *itr;
        }
        // cv::normalize( hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
        for(cv::SparseMatIterator_<float>
              itr(hist.begin<float>()),
              itr_end(hist.end<float>()); itr!=itr_end; ++itr)
          if(*itr>255)  *itr= 255;
        // std::cerr<<"==="<<i;
        // for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
        // std::cerr<<" "<<hist.at<float>(h,s)<<" "<<hist_tmp.at<float>(h,s);
        // std::cerr<<std::endl;
      }

      cv::MatND backproj;
      cv::calcBackProject(&frame_hsv, 1, channels, hist, backproj, ranges, 1, true);


      // Visualize background subtraction
      frame_masked= 0.3*frame;
      cv::Mat masks[3]= {mask, 0.5*mask, 0.0*mask}, cmask;
      cv::merge(masks,3,cmask);
      frame_masked+= cmask;
      // cv::imshow(window, frame);
      cv::imshow("BkgSbtr", frame_masked);
      // cv::imshow("mask", mask);

      cv::imshow("BackProject", backproj);
    }

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  running= !running;
    else if(c=='r')
    {
      hist.clear();
    }
    else if(c=='d')  detecting= !detecting;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
