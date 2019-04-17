/* Compile:
x++ color_detector.cpp -- -lopencv_core -lopencv_imgproc -lopencv_highgui

NOTE: the difference from version 1:
- Added Gaussian blur and dilate/erode to remove the noise.
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
#include <cassert>

/*! Detect specific color from an input image, and return the mask image (0 or 255).
  \param src_img     Input image.
  \param color_code  Color conversion code where the colors are detected, like CV_BGR2HSV, CV_BGR2Lab.
  \param lower       Lower bound of color.
  \param upper       Upper bound of color.
  \note  If lower[i]<=upper[i], lower[i]<=color<=upper[i] is detected.  Otherwise, i.e. when lower[i]>upper[i], color<=lower[i] or upper[i]<=color is detected.

  The algorithm is based on: http://imagingsolution.blog107.fc2.com/blog-entry-248.html
*/
cv::Mat ColorDetector(const cv::Mat &src_img,
  int color_code, const cv::Vec3b &lower, const cv::Vec3b &upper,
  bool using_blur=true, int dilations_erosions=2)
{
  int i, k;

  assert(src_img.type()==CV_8UC3);

  cv::Mat lut;

  cv::Mat color_img;
  if(using_blur)
  {
    GaussianBlur(src_img, color_img, cv::Size(7,7), 2.5, 2.5);
    cv::cvtColor(color_img, color_img, color_code);  // Color conversion with code
  }
  else
    cv::cvtColor(src_img, color_img, color_code);  // Color conversion with code

  // Making a lookup table of 3 channels
  lut.create(256, 1, CV_8UC3);

  for (i = 0; i < 256; i++)
  {
    cv::Vec3b val;
    for (k = 0; k < 3; k++)
    {
      if (lower[k] <= upper[k])
      {
        if ((lower[k] <= i) && (i <= upper[k]))
          val[k] = 255;
        else
          val[k] = 0;
      }
      else
      {
        if ((i <= upper[k]) || (lower[k] <= i))
          val[k] = 255;
        else
          val[k] = 0;
      }
    }
    lut.at<cv::Vec3b>(i,0)= val;
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
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    cv::imshow("camera", frame);
    // Human skin:
    // mask_img= ColorDetector(frame,CV_BGR2HSV, cv::Vec3b(0,80,0), cv::Vec3b(10,255,255));
    // mask_img= ColorDetector(frame,CV_BGR2HSV, cv::Vec3b(177,80,102), cv::Vec3b(22,131,200));
    // Nexus' red:
    mask_img= ColorDetector(frame,CV_BGR2HSV, cv::Vec3b(161,80,83), cv::Vec3b(25,188,206));

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
