/* Compile:
x++ color_detector.cpp -- -lopencv_core -lopencv_imgproc -lopencv_highgui

NOTE: the difference from version 1:
- Added Gaussian blur and dilate/erode to remove the noise.
NOTE: the difference from version 1 & 2:
- Now, it can detects multiple colors.
*/

// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>  // cvtColor
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

/*! Detect specific colors from an input image, and return the mask image (0 or 255).
  \param src_img     Input image.
  \param color_code  Color conversion code where the colors are detected, like CV_BGR2HSV, CV_BGR2Lab.
  \param colors      Colors to be detected.
  \param col_radius  Threshold of each color.  If col_radius[i]<0, we treate as 0==256 (i.e. cyclic).
*/
cv::Mat ColorDetector(const cv::Mat &src_img,
  int color_code, const std::vector<cv::Vec3b> &colors, const cv::Vec3s &col_radius,
  bool using_blur=true, int dilations_erosions=2)
{
  assert(src_img.type()==CV_8UC3);

  cv::Mat color_img;
  if(using_blur)
  {
    GaussianBlur(src_img, color_img, cv::Size(7,7), 2.5, 2.5);
    cv::cvtColor(color_img, color_img, color_code);  // Color conversion with code
  }
  else
    cv::cvtColor(src_img, color_img, color_code);  // Color conversion with code

  // TODO  seperate it into a module, do not calculate unless the colors are updated
  // Making a lookup table of 3 channels
  cv::Mat lut(256, 1, CV_8UC3);
  lut= cv::Scalar(0,0,0);

  for(std::vector<cv::Vec3b>::const_iterator citr(colors.begin()),clast(colors.end()); citr!=clast; ++citr)
  {
    for(int k(0); k<3; ++k)
    {
      int crad(std::abs(col_radius[k]));
      for(int i(-crad); i<=+crad; ++i)
      {
        int idx((*citr)[k]+i);
        if(col_radius[k]>=0)
        {
          if(idx<0 || 255<idx)  continue;
        }
        else
        {
          while(idx<0)  idx+= 256;
          idx= (idx%256);
        }
        lut.at<cv::Vec3b>(idx,0)[k]= 255;
      }
    }
  }

  // Apply the lookup table (for each channel, the image is binarized)
  cv::LUT(color_img, lut, color_img);

  cv::Mat ch_imgs[3];
  cv::split(color_img, ch_imgs);

  // For each pixel, take "and" operation between all channels
  cv::Mat mask_img;
  cv::bitwise_and(ch_imgs[0], ch_imgs[1], mask_img);
  cv::bitwise_and(mask_img, ch_imgs[2], mask_img);

  if(dilations_erosions>0)
  {
    cv::dilate(mask_img,mask_img,cv::Mat(),cv::Point(-1,-1),dilations_erosions);
    cv::erode(mask_img,mask_img,cv::Mat(),cv::Point(-1,-1),dilations_erosions);
  }

  return mask_img;
}



std::vector<cv::Vec3b>  detect_colors;
void OnMouse(int event, int x, int y, int, void *vpimg)
{
  if(event == cv::EVENT_RBUTTONDOWN)
  {
    detect_colors.clear();
    return;
  }

  if(event != cv::EVENT_LBUTTONDOWN)
    return;

  cv::Mat *pimg(reinterpret_cast<cv::Mat*>(vpimg));
  cv::Mat original(1,1,pimg->type()), converted;
  original.at<cv::Vec3b>(0,0)= pimg->at<cv::Vec3b>(y,x);  // WARNING: be careful about the order of y and x
  cv::cvtColor(original, converted, CV_BGR2HSV);
  std::cout<< "BGR: "<<original.at<cv::Vec3b>(0,0)<<"  HSV: "<<converted.at<cv::Vec3b>(0,0)<<std::endl;
  detect_colors.push_back(converted.at<cv::Vec3b>(0,0));
}

int main(int argc, char **argv)
{
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  cv::namedWindow("camera",1);
  cv::namedWindow("detected",1);
  cv::namedWindow("mask_img",1);
  cv::Mat frame, mask_img, detected;

  // std::vector<cv::Vec3b>  detect_colors;
  // detect_colors.push_back(cv::Vec3b(200,100,200));
  cv::setMouseCallback("camera", OnMouse, &frame);

  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);
    // mask_img= ColorDetector(frame,CV_BGR2HSV, detect_colors, cv::Vec3s(-20,50,50));
    mask_img= ColorDetector(frame,CV_BGR2HSV, detect_colors, cv::Vec3s(-3,5,5));

    int nonzero= cv::countNonZero(mask_img);
    std::cout<<"nonzero: "<<nonzero<<" / "<<mask_img.total()<<std::endl;
    cv::imshow("mask_img", mask_img);

    // Apply the mask image
    detected.create(frame.rows, frame.cols, CV_8UC3);
    detected= cv::Scalar(0,0,0);
    frame.copyTo(detected, mask_img);

    cv::imshow("detected", detected);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
