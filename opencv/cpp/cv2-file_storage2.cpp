//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage2.cpp
    \brief   YAML read/write test
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    May.08, 2016

g++ -g -Wall -O2 -o cv2-file_storage2.out cv2-file_storage2.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TCameraInfo
{
  std::string DevID;
  int Width, Height;
  std::string PixelFormat;
  int Rotate90n;
};

void Print(const std::vector<TCameraInfo> &cam_info)
{
  int i(0);
  for(std::vector<TCameraInfo>::const_iterator itr(cam_info.begin()),itr_end(cam_info.end()); itr!=itr_end; ++itr,++i)
  {
    std::cout<<"No. "<<i<<std::endl;
    #define WRITE_VAR(x)  std::cout<<"  "#x": "<<itr->x<<std::endl;
    WRITE_VAR(DevID       );
    WRITE_VAR(Width       );
    WRITE_VAR(Height      );
    WRITE_VAR(PixelFormat );
    WRITE_VAR(Rotate90n   );
    #undef WRITE_VAR
  }
}

void WriteToYAML(const std::vector<TCameraInfo> &cam_info, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::WRITE);
  fs<<"CameraInfo"<<"[";
  for(std::vector<TCameraInfo>::const_iterator itr(cam_info.begin()),itr_end(cam_info.end()); itr!=itr_end; ++itr)
  {
    #define WRITE_VAR(x)  fs<<#x<<itr->x;
    fs<<"{";
    WRITE_VAR(DevID       );
    WRITE_VAR(Width       );
    WRITE_VAR(Height      );
    WRITE_VAR(PixelFormat );
    WRITE_VAR(Rotate90n   );
    fs<<"}";
    #undef WRITE_VAR
  }
  fs<<"]";
  fs.release();
}

void ReadFromYAML(std::vector<TCameraInfo> &cam_info, const std::string &file_name)
{
  cam_info.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["CameraInfo"];
  for(cv::FileNodeIterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
  {
    TCameraInfo cf;
    #define READ_VAR(x)  (*itr)[#x]>>cf.x;
    READ_VAR(DevID       );
    READ_VAR(Width       );
    READ_VAR(Height      );
    READ_VAR(PixelFormat );
    READ_VAR(Rotate90n   );
    #undef READ_VAR
    cam_info.push_back(cf);
  }
  fs.release();
}

int main(int argc, char**argv)
{
  std::vector<TCameraInfo> cam_info1;
  TCameraInfo cf;
  cf.DevID= "1";
  cf.Width= 640;
  cf.Height= 480;
  cf.PixelFormat= "MJPG";
  cf.Rotate90n= 0;
  cam_info1.push_back(cf);
  cf.DevID= "2";
  cam_info1.push_back(cf);
  cf.DevID= "3";
  cf.Rotate90n= 2;
  cam_info1.push_back(cf);
  cf.DevID= "4";
  cam_info1.push_back(cf);

  std::cout<<"###cam_info1:"<<std::endl;
  Print(cam_info1);
  WriteToYAML(cam_info1, "/tmp/cam_info1.yaml");
  std::cout<<"###cam_info1: wrote to /tmp/cam_info1.yaml"<<std::endl;

  std::vector<TCameraInfo> cam_info2;
  ReadFromYAML(cam_info2, "/tmp/cam_info1.yaml");
  std::cout<<"###cam_info2: read from /tmp/cam_info1.yaml"<<std::endl;
  std::cout<<"###cam_info2:"<<std::endl;
  Print(cam_info2);

  return 0;
}
//-------------------------------------------------------------------------------------------
