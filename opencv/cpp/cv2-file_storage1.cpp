//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Apr.01, 2016

g++ -g -Wall -O2 -o cv2-file_storage1.out cv2-file_storage1.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat mA,mB;
  cv::Vec3b v3b(1,2,3);
  cv::Vec3d v3d(1.1,2.2,3.3);
  double x(1.234);
  mA= (cv::Mat_<unsigned char>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  mB= (cv::Mat_<double>(3,1)<<1.1,2.2,3.3);

  {
    cv::FileStorage fs("/tmp/cv2_test.yaml", cv::FileStorage::WRITE);
    #define PROC_VAR(x)  fs<<#x<<(x);
    PROC_VAR( mA  )
    PROC_VAR( mB  )
    PROC_VAR( v3b )
    PROC_VAR( v3d )
    PROC_VAR( x   )
    #undef PROC_VAR
    fs.release();
  }
  {
    cv::FileStorage fs("/tmp/cv2_test.yaml", cv::FileStorage::READ);
    #define PROC_VAR(x)  if(!fs[#x].empty())  fs[#x] >> x;
    PROC_VAR( mA  )
    PROC_VAR( mB  )
    PROC_VAR( v3b )
    PROC_VAR( v3d )
    PROC_VAR( x   )
    #undef PROC_VAR
    fs.release();
  }
  {
    #define PROC_VAR(x)  std::cout<<#x"= "<<(x)<<std::endl;
    PROC_VAR( mA  )
    PROC_VAR( mB  )
    PROC_VAR( v3b )
    PROC_VAR( v3d )
    PROC_VAR( x   )
    #undef PROC_VAR
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
