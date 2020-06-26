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

#include <fstream>
/*! \brief check the filename exists */
inline bool FileExists(const std::string &filename)
{
  bool res(false);
  std::ifstream ifs (filename.c_str());
  res = ifs.is_open();
  ifs.close();
  return res;
}
//-------------------------------------------------------------------------------------------

namespace cv
{

// Define a new bool reader in order to accept "true/false"-like values.
void read_bool(const cv::FileNode &node, bool &value, const bool &default_value)
{
  std::string s(static_cast<std::string>(node));
  if(s=="y"||s=="Y"||s=="yes"||s=="Yes"||s=="YES"||s=="true"||s=="True"||s=="TRUE"||s=="on"||s=="On"||s=="ON")
    {value=true; return;}
  if(s=="n"||s=="N"||s=="no"||s=="No"||s=="NO"||s=="false"||s=="False"||s=="FALSE"||s=="off"||s=="Off"||s=="OFF")
    {value=false; return;}
  value= static_cast<int>(node);
}
// Specialize cv::operator>> for bool.
template<> inline void operator >> (const cv::FileNode& n, bool& value)
{
  read_bool(n, value, false);
}

//-------------------------------------------------------------------------------------------
} // cv
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat mA,mB;
  cv::Vec3b v3b(1,2,3);
  cv::Vec3d v3d(1.1,2.2,3.3);
  double x(1.234);
  bool b0(false), b1(true);
  mA= (cv::Mat_<unsigned char>(3,3)<<1,0,1, 0,0,1, 0,1,1);
  mB= (cv::Mat_<double>(3,1)<<1.1,2.2,3.3);

  if(!FileExists("/tmp/cv2_test.yaml"))
  {
    cv::FileStorage fs("/tmp/cv2_test.yaml", cv::FileStorage::WRITE);
    #define PROC_VAR(x)  fs<<#x<<(x);
    PROC_VAR( mA  )
    PROC_VAR( mB  )
    PROC_VAR( v3b )
    PROC_VAR( v3d )
    PROC_VAR( x   )
    PROC_VAR( b0  )
    PROC_VAR( b1  )
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
    PROC_VAR( b0  )
    PROC_VAR( b1  )
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
    PROC_VAR( b0  )
    PROC_VAR( b1  )
    #undef PROC_VAR
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
