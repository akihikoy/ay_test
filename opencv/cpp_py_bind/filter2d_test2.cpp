//-------------------------------------------------------------------------------------------
/*! \file    filter2d_test2.cpp
    \brief   Test of making a filer function for Python that takes numpy.array as the parameters.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Sep.26, 2023

g++ -g -Wall -O3 -o filter2d_test2.so filter2d_test2.cpp -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_video -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lboost_thread -lboost_system -shared -fPIC -I/usr/include/python2.7 `python -m pybind11 --includes`

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <unistd.h>
//-------------------------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//-------------------------------------------------------------------------------------------

cv::Mat Kernel;

void MakeKernel(int sx, int sy)
{
  Kernel= cv::Mat::ones(cv::Size(sx,sy),CV_32F)/(float)(sx*sy);
}
//-------------------------------------------------------------------------------------------

void ApplyFilter(const cv::Mat &src, cv::Mat &dst)
{
  cv::filter2D(src, dst, /*ddepth=*/-1, Kernel);
}
//-------------------------------------------------------------------------------------------

namespace py= pybind11;

py::array_t<unsigned char> MatToNumpy(const cv::Mat &image)
{
  int h= image.rows;
  int w= image.cols;
  int c= image.channels();
  return py::array_t<unsigned char>({ h, w, c }, image.data);
}

cv::Mat NumpyToMat(const py::array_t<unsigned char> &input_array)
{
  py::buffer_info buf_info(input_array.request());
  int h= buf_info.shape[0];
  int w= buf_info.shape[1];
  int c= buf_info.shape[2];
  return cv::Mat(h, w, c == 3 ? CV_8UC3 : CV_8UC1, buf_info.ptr);
}

py::array_t<unsigned char> ApplyFilter(const py::array_t<unsigned char> &src)
{
  cv::Mat src_cv, dst_cv;
  src_cv= NumpyToMat(src);
  ApplyFilter(src_cv, dst_cv);
  // cv::imshow("dst_cv", dst_cv);
  return MatToNumpy(dst_cv);
}
//-------------------------------------------------------------------------------------------

PYBIND11_MODULE(filter2d_test2, m)
{
  m.doc() = "filter2d_test2.";
  m.def("MakeKernel", &MakeKernel, "MakeKernel");
  // NOTE: For a overloaded function like ApplyFilter, we need a cast.
  m.def("ApplyFilter", static_cast<py::array_t<unsigned char> (*)(const py::array_t<unsigned char> &)>(&ApplyFilter), "ApplyFilter",
        py::arg("src") );
}
//-------------------------------------------------------------------------------------------
