//-------------------------------------------------------------------------------------------
/*! \file    filter2d_test1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.19, 2022

NOTE: This program is incomplete (does not work).

Compile with main:
g++ -g -Wall -O2 -o filter2d_test1.out filter2d_test1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

Compile as a shared object (comment out main):
g++ -g -Wall -O2 -shared -fPIC -o filter2d_test1.so filter2d_test1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/python2.7 `python -m pybind11 --includes`
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
//-------------------------------------------------------------------------------------------
namespace trick
{
}
//-------------------------------------------------------------------------------------------

class CV_EXPORTS_W TFilter2DTest1
{
public:
  CV_PROP cv::Mat kernel;
  CV_WRAP TFilter2DTest1(int sx=1, int sy=100)
      : kernel(cv::Size(sx,sy),CV_32F)
    {
      kernel= cv::Mat::ones(kernel.size(),CV_32F)/(float)(kernel.rows*kernel.cols);
    }
  CV_WRAP void Apply(cv::InputArray src, cv::OutputArray dst)
    {
      cv::filter2D(src, dst, /*ddepth=*/-1, kernel);
    }
private:
};
//-------------------------------------------------------------------------------------------

#if 0
int main(int argc, char**argv)
{
  cv::VideoCapture cap((argc>1)?atoi(argv[1]):0);

  TFilter2DTest1 filter;
  cv::namedWindow("camera",1);
  cv::Mat frame, filtered;
  for(;;)
  {
    cap >> frame; // get a new frame from camera

    filter.Apply(frame, filtered);

    cv::imshow("camera", filtered);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }
  return 0;
}
//-------------------------------------------------------------------------------------------
#else
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py= pybind11;
PYBIND11_PLUGIN(filter2d_test1)
{
  py::module m("filter2d_test1", "filter2d_test1");
  py::class_<TFilter2DTest1>(m, "TFilter2DTest1")
    .def(py::init<int,int>())
    .def("Apply", &TFilter2DTest1::Apply);
}
//-------------------------------------------------------------------------------------------
#endif
