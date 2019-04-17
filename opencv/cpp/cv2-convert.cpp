// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
#endif
#include <iostream>
// g++ -I -Wall cv2-convert.cpp -o cv2-convert -I/usr/include/opencv2 -lopencv_core -lopencv_ml -lopencv_video -lopencv_legacy -lopencv_imgproc -lopencv_highgui

#define print(var) std::cout<<#var"= "<<(var)<<std::endl

int main(int argc, char **argv)
{
#if 1
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
  cv::namedWindow("converted",1);
  cv::namedWindow("gray",1);
  cv::Mat frame,converted,gray;

  cap >> frame;
  // converted.create(frame.rows,frame.cols,CV_8UC3);
  converted.create(frame.rows,frame.cols,CV_32FC3);
  // converted.create(frame.rows,frame.cols,CV_8UC1);  // Will not be converted to gray scale
  for(;;)
  {
    cap >> frame; // get a new frame from camera
    frame.convertTo(converted,converted.type());
    cv::cvtColor(frame,gray,CV_BGR2GRAY);
    cv::imshow("camera", frame);
    cv::imshow("converted", converted/255.0);
    cv::imshow("gray", gray);
    // cv::imshow("converted", converted);
    if(cv::waitKey(10) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
#endif
#if 0
  CvCapture *capture = 0;
  IplImage *frame = 0;
  IplImage *converted = 0;
  capture= cvCreateCameraCapture (0);
  cvNamedWindow ("camera", CV_WINDOW_AUTOSIZE);
  cvNamedWindow ("converted", CV_WINDOW_AUTOSIZE);

  for(;;)
  {
    frame = cvQueryFrame (capture);
    if (frame==NULL)
    {
      std::cerr<<"missing captured frame"<<std::endl;
      continue;
    }
    if (converted==NULL)
      // converted= cvCreateImage (cvGetSize (frame), IPL_DEPTH_8U, 3);
      converted= cvCreateImage (cvGetSize (frame), IPL_DEPTH_32F, 3);
    cvConvertScale(frame,converted);
    cvShowImage ("camera", frame);
    cvShowImage ("converted", converted);
    if(cv::waitKey(10) >= 0) break;
  }
  cvReleaseImage (&frame);
  cvReleaseImage (&converted);
  cvDestroyWindow ("camera");
  cvDestroyWindow ("converted");
  return 0;
#endif
}
