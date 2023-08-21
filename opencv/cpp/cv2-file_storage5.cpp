//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage5.cpp
    \brief   Example of reading configuration from a YAML text.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.28, 2023

g++ -g -Wall -O2 -o cv2-file_storage5.out cv2-file_storage5.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
//-------------------------------------------------------------------------------------------
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TConfig
{
  std::string pkg_dir           ;
  std::string cam_config        ;
  std::string blobtrack_config  ;
  std::string objdettrack_config;
  std::string blob_calib_prefix ;
  std::string vout_base         ;
  int         frame_skip        ;
  int         target_fps        ;
  int         capture_fps       ;
  bool        camera_auto_reopen;
};

void ReadFromYAML(TConfig &config, const std::string &file_name)
{
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  cv::FileNode data= fs["Config"];
  print(data.empty());
  if(data.empty())
  {
    fs.release();
    return;
  }
  #define PROC_VAR(v)  if(!data[#v].empty()) data[#v]>>config.v
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
  fs.release();
}
//-------------------------------------------------------------------------------------------

#ifndef LIBRARY
int main(int argc, char**argv)
{
  TConfig config;
//   config.pkg_dir           = ".";
//   config.cam_config        = "config/cam1.yaml";
//   config.blobtrack_config  = "config/cam1.yaml";
//   config.objdettrack_config= "config/cam1.yaml";
//   config.blob_calib_prefix = "blob_";
//   config.vout_base         = "/tmp/vout-";
//   config.frame_skip        = 0;
//   config.target_fps        = 0;
//   config.capture_fps       = 0;
//   config.camera_auto_reopen= true;

  ReadFromYAML(config, "cv2-file_storage5.yaml");
//   std::string TMP_CONFIG_YAML= "/tmp/cv2-file_storage4.yaml";
//   {
//     std::ofstream ofs(TMP_CONFIG_YAML);
//     ofs<<"%YAML:1.0"<<std::endl;
//     for(int c(1); c<argc; ++c)
//       ofs<<argv[c]<<std::endl;
//   }
//   ReadFromYAML(config, TMP_CONFIG_YAML);

  print(config.pkg_dir           );
  print(config.cam_config        );
  print(config.blobtrack_config  );
  print(config.objdettrack_config);
  print(config.blob_calib_prefix );
  print(config.vout_base         );
  print(config.frame_skip        );
  print(config.target_fps        );
  print(config.capture_fps       );
  print(config.camera_auto_reopen);

  return 0;
}
#endif//LIBRARY
//-------------------------------------------------------------------------------------------
