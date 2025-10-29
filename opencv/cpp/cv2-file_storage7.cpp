//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage7.cpp
    \brief   Test to merge multiple WriteToYAML functions into a single YAML file.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.22, 2023

g++ -g -Wall -O2 -o cv2-file_storage7.out cv2-file_storage7.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#define LIBRARY
#include "cv2-file_storage5.cpp"
//-------------------------------------------------------------------------------------------
template<typename t_vec>
void PrintContainer(const t_vec &v)
{
  for(typename t_vec::const_iterator itr(v.begin()),itr_end(v.end()); itr!=itr_end; ++itr)
    std::cout<<" "<<*itr;
}
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define printv(var) std::cout<<#var"= ";PrintContainer(var);std::cout<<std::endl
//-------------------------------------------------------------------------------------------

enum TCameraKind {ckRGB=0, ckMono, ckDepth};

struct TCameraInfo
{
  std::string DevID;
  int Width, Height;
  std::string PixelFormat;
  int Rotate90n;
  TCameraKind Kind;
  std::vector<cv::Vec2f> ROI{{0.1f, 0.2}, {0.7f, 0.8f}};  // NOTE: vector<Vec2f> is serialized in YAML.
  std::vector<std::vector<float>> ROI2{{0.1f, 0.2}, {0.7f, 0.8f}};  // NOTE: vector<vector> is not serialized in YAML.
};

void WriteToYAML(const std::vector<TCameraInfo> &cam_info, const std::string &file_name, cv::FileStorage *pfs=NULL)
{
  cv::FileStorage fs1;
  if(pfs==NULL)  fs1.open(file_name, cv::FileStorage::WRITE);
  cv::FileStorage &fs(pfs==NULL?fs1:*pfs);
  fs<<"CameraInfo"<<"[";
  for(std::vector<TCameraInfo>::const_iterator itr(cam_info.begin()),itr_end(cam_info.end()); itr!=itr_end; ++itr)
  {
    #define PROC_VAR(x)  fs<<#x<<itr->x;
    fs<<"{";
    PROC_VAR(DevID       );
    PROC_VAR(Width       );
    PROC_VAR(Height      );
    PROC_VAR(PixelFormat );
    PROC_VAR(Rotate90n   );
    PROC_VAR(Kind        );
    PROC_VAR(ROI         );
    PROC_VAR(ROI2        );
    fs<<"}";
    #undef PROC_VAR
  }
  fs<<"]";
  if(pfs==NULL)  fs.release();
}
void ReadFromYAML(std::vector<TCameraInfo> &cam_info, const std::string &file_name)
{
  cam_info.clear();
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["CameraInfo"];
  for(cv::FileNodeIterator itr(data.begin()),itr_end(data.end()); itr!=itr_end; ++itr)
  {
    TCameraInfo cf;
    #define PROC_VAR(x)  (*itr)[#x]>>cf.x;
    PROC_VAR(DevID       );
    PROC_VAR(Width       );
    PROC_VAR(Height      );
    PROC_VAR(PixelFormat );
    PROC_VAR(Rotate90n   );
    PROC_VAR(Kind        );
    PROC_VAR(ROI         );
    PROC_VAR(ROI2        );
    #undef PROC_VAR
    cam_info.push_back(cf);
  }
  fs.release();
}


void WriteToYAML(const TConfig &config, const std::string &file_name, cv::FileStorage *pfs=NULL)
{
  cv::FileStorage fs1;
  if(pfs==NULL)  fs1.open(file_name, cv::FileStorage::WRITE);
  cv::FileStorage &fs(pfs==NULL?fs1:*pfs);
  fs<<"Config"<<"{";
  #define PROC_VAR(x)  fs<<#x<<config.x;
  PROC_VAR(pkg_dir           );
  PROC_VAR(cam_config        );
  PROC_VAR(blobtrack_config  );
  PROC_VAR(objdettrack_config);
  PROC_VAR(blob_calib_prefix );
  PROC_VAR(vout_base         );
  PROC_VAR(frame_skip        );
  PROC_VAR(target_fps        );
  PROC_VAR(capture_fps       );
  PROC_VAR(camera_auto_reopen);
  #undef PROC_VAR
  fs<<"}";
  if(pfs==NULL)  fs.release();
}

//-------------------------------------------------------------------------------------------


int main(int argc, char**argv)
{
  std::vector<TCameraInfo> cam_info;
  TCameraInfo cf;
  cf.DevID= "1";
  cf.Width= 640;
  cf.Height= 480;
  cf.PixelFormat= "MJPG";
  cf.Rotate90n= 0;
  cf.Kind= ckRGB;
  cam_info.push_back(cf);
  cf.DevID= "2";
  cf.Kind= ckMono;
  cam_info.push_back(cf);

  TConfig config;
  ReadFromYAML(config, "cv2-file_storage7.yaml");
  ReadFromYAML(cam_info, "cv2-file_storage7.yaml");

  std::cout<<"-------------------------"<<std::endl;
  WriteToYAML(cam_info, "/dev/stdout");
  std::cout<<"-------------------------"<<std::endl;
  WriteToYAML(config, "/dev/stdout");
  std::cout<<"-------------------------"<<std::endl;
  cv::FileStorage fs1("/dev/stdout", cv::FileStorage::WRITE);
  WriteToYAML(cam_info, "", &fs1);
  WriteToYAML(config, "", &fs1);
  fs1.release();
  std::cout<<"-------------------------"<<std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
