//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage-kp.cpp
    \brief   Test OpenCV file storage about cv::KeyPoint with different CV versions.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Nov.12, 2021

g++ -g -Wall -O2 -o cv2-file_storage-kp.out cv2-file_storage-kp.cpp -lopencv_core
set HOME2=/home/ay2 && g++ -g -Wall -O2 -o cv2-file_storage-kp.out cv2-file_storage-kp.cpp -lopencv_core -I$HOME2/.local/include -L$HOME2/.local/lib -Wl,-rpath=$HOME2/.local/lib
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
namespace cv
{

#if CV_MAJOR_VERSION<3 || (CV_MAJOR_VERSION==3 && CV_MINOR_VERSION<4)
// These functions are defined in OpenCV 3.4+.
void write(cv::FileStorage &fs, const cv::String&, const cv::KeyPoint &x)
{
  #define PROC_VAR(v)  fs<<#v<<x.v;
  fs<<"{";
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  fs<<"}";
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------
void read(const cv::FileNode &data, cv::KeyPoint &x, const cv::KeyPoint &default_value)
{
  #define PROC_VAR(v)  if(!data[#v].empty()) data[#v]>>x.v;
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------
#endif // <OpenCV 3.4


//-------------------------------------------------------------------------------------------
}  // namespace cv
//-------------------------------------------------------------------------------------------


// Common write/read for OpenCV versions:
void kp_write(cv::FileStorage &fs, const cv::String&, const cv::KeyPoint &x)
{
  #define PROC_VAR(v)  fs<<#v<<x.v;
  fs<<"{";
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  fs<<"}";
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------
void kp_read(const cv::FileNode &data, cv::KeyPoint &x, const cv::KeyPoint &default_value)
{
  #define PROC_VAR(v)  if(!data[#v].empty()) data[#v]>>x.v;
  PROC_VAR(angle);
  PROC_VAR(class_id);
  PROC_VAR(octave);
  PROC_VAR(pt);
  PROC_VAR(response);
  PROC_VAR(size);
  #undef PROC_VAR
}
//-------------------------------------------------------------------------------------------



int main(int argc, char**argv)
{
  cv::KeyPoint kp;
  kp.pt.x= 100;
  kp.pt.y= 200;

  #define STR_HELPER(x) #x
  #define STR(x) STR_HELPER(x)
  #define FILE_NAME "/tmp/kp" STR(CV_MAJOR_VERSION) "." STR(CV_MINOR_VERSION) ".yaml"

  std::cout<<"kp: "<<kp.pt<<" "<<kp.size<<" "<<kp.angle<<" "<<std::endl;
  {
    cv::FileStorage fs(FILE_NAME, cv::FileStorage::WRITE);
    fs<<"kp"<<kp;
    fs.release();
  }
  std::cout<<"saved into: "<<FILE_NAME<<std::endl;

  {
    cv::FileStorage fs(FILE_NAME, cv::FileStorage::READ);
    fs["kp"]>>kp;
    fs.release();
  }
  std::cout<<"kp: "<<kp.pt<<" "<<kp.size<<" "<<kp.angle<<" "<<std::endl;

  {
    cv::FileStorage fs("/tmp/kpX.yaml", cv::FileStorage::WRITE);
    fs<<"kp";
    kp_write(fs,"",kp);
    fs.release();
  }
  std::cout<<"saved into: "<<"/tmp/kpX.yaml"<<std::endl;

  {
    cv::FileStorage fs("/tmp/kpX.yaml", cv::FileStorage::READ);
    kp_read(fs["kp"],kp,cv::KeyPoint());
    fs.release();
  }
  std::cout<<"kp: "<<kp.pt<<" "<<kp.size<<" "<<kp.angle<<" "<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
