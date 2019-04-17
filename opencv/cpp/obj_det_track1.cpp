//-------------------------------------------------------------------------------------------
/*! \file    obj_det_track1.cpp
    \brief   Object detection and tracking
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.13, 2017

g++ -I -Wall obj_det_track1.cpp -o obj_det_track1.out -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui
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
  int history(30), n_erode1(0), n_erode2(2), n_dilate2(7), n_threshold2(150);
  int f_bg20(30), f_gain20(10);

  // Histogram
  cv::Mat hist, hist_tmp, hist_bg;
  float h_range[] = { 0, 179 };
  float s_range[] = { 0, 255 };
  int h_bins = 100; int s_bins = 10;
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
  float f_bg(1.0), f_gain(1.0);

  cv::Mat frame, frame_hsv, mask_bs, mask_bser;
  bool running(true), detecting_mode(true), calib_bg(true), calib_mode(false);
  // int n_hist(0);
  for(int i(0);;++i)
  {
    if(running)
    {
      cap >> frame; // get a new frame from camera
      if(i%2!=0)  continue;  // Adjust FPS for speed up
      cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

      if(calib_bg)
      {
        cv::calcHist(&frame_hsv, 1, channels, cv::Mat(), hist_bg, n_channels, histSize, ranges, true, false);
        calib_bg= false;
      }

      // if(detecting_mode)
      // {
      bkg_sbtr(frame, mask_bs, 1./float(history));
      // cv::threshold(mask, mask, 200, 0, cv::THRESH_TOZERO_INV);
      cv::erode(mask_bs,mask_bser,cv::Mat(),cv::Point(-1,-1), n_erode1);
      // cv::dilate(mask,mask,cv::Mat(),cv::Point(-1,-1), n_erode);
      // cv::erode(mask,mask,cv::Mat(),cv::Point(-1,-1), n_erode+1);

      // if(i>1)
      // {
        // std::cerr<<"---"<<i;
        // for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
        // std::cerr<<" "<<hist.at<float>(h,s);
        // std::cerr<<std::endl;
      // }

      if(detecting_mode)
      {
        f_bg= float(f_bg20)/20.0;
        f_gain= float(f_gain20)/20.0;

        // Get the Histogram and normalize it
        cv::calcHist(&frame_hsv, 1, channels, mask_bser, hist_tmp, n_channels, histSize, ranges, true, false);
        // Add to the object model
        if(i==0)
        {
          // hist_tmp.copyTo(hist);
          // #if N_CHANNELS==2
          // for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
            // hist.at<float>(h,s)= f_gain*std::max(0.0f, hist.at<float>(h,s)-f_bg*hist_bg.at<float>(h,s));
          // #elif N_CHANNELS==1
          // for(int h(0);h<h_bins;++h)
            // hist.at<float>(h)= f_gain*std::max(0.0f, hist.at<float>(h)-f_bg*hist_bg.at<float>(h));
          // #endif
// std::cerr<<hist_tmp.rows<<" "<<hist_tmp.cols<<" "<<hist_tmp.channels()<<" "<<hist_tmp.size()<<std::endl;
          hist.create(hist_tmp.size(), hist_tmp.type());
        }
        // else
        // {
        // ++n_hist;
        // float alpha= 1.0/float(n_hist);
        // float alpha= 0.1;
        #if N_CHANNELS==2
        for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
          hist.at<float>(h,s)+= f_gain*std::max(0.0f,hist_tmp.at<float>(h,s)-f_bg*hist_bg.at<float>(h,s));
        #elif N_CHANNELS==1
        for(int h(0);h<h_bins;++h)
          hist.at<float>(h)+= f_gain*std::max(0.0f,hist_tmp.at<float>(h)-f_bg*hist_bg.at<float>(h));
        #endif
        // }

        // cv::normalize( hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
        #if N_CHANNELS==2
        for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
          if(hist.at<float>(h,s)>255)  hist.at<float>(h,s)= 255;
        #elif N_CHANNELS==1
        for(int h(0);h<h_bins;++h)
          if(hist.at<float>(h)>255)  hist.at<float>(h)= 255;
        #endif
        // std::cerr<<"==="<<i;
        // for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
        // std::cerr<<" "<<hist.at<float>(h,s)<<" "<<hist_tmp.at<float>(h,s);
        // std::cerr<<std::endl;
      }

      cv::Mat backproj;
      cv::calcBackProject(&frame_hsv, 1, channels, hist, backproj, ranges, 1, true);
      cv::Mat mask_obj, mask_mv;
      backproj.copyTo(mask_obj);
      // std::cerr<<mask_obj<<std::endl;
      cv::threshold(mask_obj, mask_obj, n_threshold2, 0, cv::THRESH_TOZERO);
      cv::erode(mask_obj,mask_obj,cv::Mat(),cv::Point(-1,-1), n_erode2);
      cv::dilate(mask_obj,mask_obj,cv::Mat(),cv::Point(-1,-1), n_dilate2);
      mask_bs.copyTo(mask_mv, mask_obj);


      // Visualize background subtraction
      cv::Mat disp1= 0.3*frame;
      cv::Mat masks1[3]= {mask_bser, 0.0*mask_bs, mask_bs}, cmask1;
      cv::merge(masks1,3,cmask1);
      disp1+= cmask1;
      // cv::imshow(window, frame);
      cv::imshow("BkgSbtr", disp1);
      // cv::imshow("mask", mask);

      cv::Mat disp2= 0.3*frame;
      cv::Mat masks2[3]= {mask_obj, 0.0*mask_mv, mask_mv}, cmask2;
      cv::merge(masks2,3,cmask2);
      disp2+= cmask2;
      cv::imshow("BackProject", disp2);
    }

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    if(c==' ')  running= !running;
    else if(c=='r')
    {
      // n_hist= 0;
      #if N_CHANNELS==2
      for(int h(0);h<h_bins;++h) for(int s(0);s<s_bins;++s)
        hist.at<float>(h,s)= 0;
      #elif N_CHANNELS==1
      for(int h(0);h<h_bins;++h)
        hist.at<float>(h)= 0;
      #endif
    }
    else if(c=='d')  detecting_mode= !detecting_mode;
    else if(c=='b')  calib_bg= true;
    else if(c=='C' || c=='c')
    {
      calib_mode= !calib_mode;
      if(calib_mode)
      {
        cv::createTrackbar( "History:", "BkgSbtr", &history, 100, NULL);
        cv::createTrackbar( "20*f_bg:", "BkgSbtr", &f_bg20, 100, NULL);
        cv::createTrackbar( "20*f_gain:", "BkgSbtr", &f_gain20, 100, NULL);
        cv::createTrackbar( "N-Erode(1):", "BkgSbtr", &n_erode1, 10, NULL);
        cv::createTrackbar( "N-Erode(2):", "BkgSbtr", &n_erode2, 20, NULL);
        cv::createTrackbar( "N-Dilate(2):", "BkgSbtr", &n_dilate2, 20, NULL);
        cv::createTrackbar( "Threshold(2):", "BkgSbtr", &n_threshold2, 255, NULL);
      }
      else
      {
        // Remove trackbars from window.
        cv::destroyWindow("BkgSbtr");
        cv::namedWindow("BkgSbtr",1);
      }
    }
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
