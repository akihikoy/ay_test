/* Compile:
g++ -g -Wall -O2 -o color_detector4.out color_detector4.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_videoio

NOTE: the difference from version 1:
- Added Gaussian blur and dilate/erode to remove the noise.
NOTE: the difference from version 1 & 2:
- Now, it can detects multiple colors.
NOTE: the difference from version 1 & 2 & 3:
- Class version, where some computations are optimized.
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

class TColorDetector
{
public:
  TColorDetector()
    : color_code_            (CV_BGR2HSV),
      using_blur_            (true),
      gaussian_kernel_size_  (7,7),
      gaussian_sigma_x_      (2.5),
      gaussian_sigma_y_      (2.5),
      dilations_erosions_    (2),
      lookup_table_          (256, 1, CV_8UC3)
    {
      lookup_table_= cv::Scalar(0,0,0);
    }
  ~TColorDetector()  {}

  /*! Setup colors to be detected.
    \param colors      Colors to be detected.
    \param col_radius  Threshold of each color.  If col_radius[i]<0, we treate as 0==256 (i.e. cyclic).  */
  void SetupColors(const std::vector<cv::Vec3b> &colors, const cv::Vec3s &col_radius);

  /*! Detect specific colors from the source image, and return the mask image (0 or 255).
    \param src_img  Input image.  */
  cv::Mat Detect(const cv::Mat &src_img);


  int ColorCode() const {return color_code_;}
  bool UsingBlur() const {return using_blur_;}
  const cv::Size& GaussianKernelSize() const {return gaussian_kernel_size_;}
  const double& GaussianSigmaX() const {return gaussian_sigma_x_;}
  const double& GaussianSigmaY() const {return gaussian_sigma_y_;}
  int DilationsErosions() const {return dilations_erosions_;}

  void SetColorCode(int v)  {color_code_= v;}
  void SetUsingBlur(bool v)  {using_blur_= v;}
  void SetGaussianKernelSize(const cv::Size &v)  {gaussian_kernel_size_= v;}
  void SetGaussianSigmaY(const double &v)  {gaussian_sigma_y_= v;}
  void SetGaussianSigmaX(const double &v)  {gaussian_sigma_x_= v;}
  void SetDilationsErosions(int v)  {dilations_erosions_= v;}

protected:

  // Parameters

  //! Color conversion code where the colors are detected, like CV_BGR2HSV, CV_BGR2Lab.
  int color_code_;
  //! If true, the Gaussian blur is applied to the input image.
  bool using_blur_;
  //! Kernel size of the Gaussian blur.
  cv::Size gaussian_kernel_size_;
  //! X-std dev of the Gaussian blur.
  double gaussian_sigma_x_;
  //! Y-std dev of the Gaussian blur.
  double gaussian_sigma_y_;
  //! Size of dilations/erosions.
  int dilations_erosions_;


  cv::Mat lookup_table_;

};

/*! Setup colors to be detected.
  \param colors      Colors to be detected.
  \param col_radius  Threshold of each color.  If col_radius[i]<0, we treate as 0==256 (i.e. cyclic).  */
void TColorDetector::SetupColors(const std::vector<cv::Vec3b> &colors, const cv::Vec3s &col_radius)
{
  // Making a lookup table of 3 channels
  lookup_table_.create(256, 1, CV_8UC3);
  lookup_table_= cv::Scalar(0,0,0);

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
        lookup_table_.at<cv::Vec3b>(idx,0)[k]= 255;
      }
    }
  }
}

/*! Detect specific colors from the source image, and return the mask image (0 or 255).
  \param src_img  Input image.  */
cv::Mat TColorDetector::Detect(const cv::Mat &src_img)
{
  assert(src_img.type()==CV_8UC3);

  cv::Mat color_img;
  if(using_blur_)
  {
    GaussianBlur(src_img, color_img, gaussian_kernel_size_, gaussian_sigma_x_, gaussian_sigma_y_);
    cv::cvtColor(color_img, color_img, color_code_);  // Color conversion with code
  }
  else
    cv::cvtColor(src_img, color_img, color_code_);  // Color conversion with code

  // Apply the lookup table (for each channel, the image is binarized)
  cv::LUT(color_img, lookup_table_, color_img);

  cv::Mat ch_imgs[3];
  cv::split(color_img, ch_imgs);

  // For each pixel, take "and" operation between all channels
  cv::Mat mask_img;
  cv::bitwise_and(ch_imgs[0], ch_imgs[1], mask_img);
  cv::bitwise_and(mask_img, ch_imgs[2], mask_img);

  if(dilations_erosions_>0)
  {
    cv::dilate(mask_img,mask_img,cv::Mat(),cv::Point(-1,-1), dilations_erosions_);
    cv::erode(mask_img,mask_img,cv::Mat(),cv::Point(-1,-1), dilations_erosions_);
  }

  return mask_img;
}



std::vector<cv::Vec3b>  detect_colors;
cv::Vec3s  col_radius(-3,5,5);
TColorDetector  col_detector;
void OnMouse(int event, int x, int y, int, void *vpimg)
{
  if(event == cv::EVENT_RBUTTONDOWN)
  {
    detect_colors.clear();
    col_detector.SetupColors(detect_colors, col_radius);
    return;
  }

  if(event != cv::EVENT_LBUTTONDOWN)
    return;

  cv::Mat *pimg(reinterpret_cast<cv::Mat*>(vpimg));
  cv::Mat original(1,1,pimg->type()), converted;
  original.at<cv::Vec3b>(0,0)= pimg->at<cv::Vec3b>(y,x);  // WARNING: be careful about the order of y and x
  cv::cvtColor(original, converted, col_detector.ColorCode());
  std::cout<< "BGR: "<<original.at<cv::Vec3b>(0,0)<<"  HSV: "<<converted.at<cv::Vec3b>(0,0)<<std::endl;
  detect_colors.push_back(converted.at<cv::Vec3b>(0,0));
  col_detector.SetupColors(detect_colors, col_radius);
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
    mask_img= col_detector.Detect(frame);

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
    if(c=='p')
    {
      for(std::vector<cv::Vec3b>::const_iterator itr(detect_colors.begin()),itr_end(detect_colors.end());
          itr!=itr_end; ++itr)
        std::cout<<" "<<*itr;
      std::cout<<std::endl;
    }
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
