#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
// g++ -g -Wall -O2 -o cv2-edge.out cv2-edge.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui

int main(int, char**)
{
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
    {
      std::cerr<<"no camera!"<<std::endl;
      return -1;
    }

    cv::Mat edges;
    cv::namedWindow("edges",1);
    for(;;)
    {
      cv::Mat frame;
      cap >> frame; // get a new frame from camera
      cv::cvtColor(frame, edges, CV_BGR2GRAY);
      cv::GaussianBlur(edges, edges, cv::Size(7,7), 1.5, 1.5);
      cv::Canny(edges, edges, 0, 30, 3);
      imshow("edges", edges);
      if(cv::waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
