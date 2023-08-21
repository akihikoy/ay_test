//-------------------------------------------------------------------------------------------
/*! \file    cv2-file_storage6.cpp
    \brief   Test of read from multiple YAML files.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.21, 2023

g++ -g -Wall -O2 -o cv2-file_storage6.out cv2-file_storage6.cpp -lopencv_core
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
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

#define LIBRARY
#include "cv2-file_storage5.cpp"

// Template to read from multiple YAML files.
// Assuming that ReadFromYAML(t_data &data, const std::string &file_name) is defined,
// and ReadFromYAML does not reset (clear) data when reading.
template<typename t_data>
void ReadFromYAML(t_data &data, const std::vector<std::string> &file_names)
{
  for(std::vector<std::string>::const_iterator itr(file_names.begin()),itr_end(file_names.end());
      itr!=itr_end; ++itr)
    ReadFromYAML(data, *itr);
}
//-------------------------------------------------------------------------------------------

void ReadTest(const std::vector<std::string> &file_names)
{
  TConfig config;

  std::cout<<"###Reading from:"<<std::endl;
  printv(file_names);
  ReadFromYAML(config, file_names);

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
  std::cout<<""<<std::endl;
}

int main(int argc, char**argv)
{
  std::vector<std::string> file_names;

  // std::cout<<"###DEBUG:"<<std::endl;
  // file_names.push_back(std::string("cv2-file_storage5.yaml"));
  // printv(file_names);
  // print(file_names.size());
  // print(file_names[0]);

  file_names.clear();
  file_names.push_back(std::string("cv2-file_storage5.yaml"));
  ReadTest(file_names);

  file_names.clear();
  file_names.push_back(std::string("cv2-file_storage6.yaml"));
  ReadTest(file_names);

  file_names.clear();
  file_names.push_back(std::string("cv2-file_storage5.yaml"));
  file_names.push_back(std::string("cv2-file_storage6.yaml"));
  ReadTest(file_names);

  file_names.clear();
  file_names.push_back(std::string("cv2-file_storage6.yaml"));
  file_names.push_back(std::string("cv2-file_storage5.yaml"));
  ReadTest(file_names);

  return 0;
}
//-------------------------------------------------------------------------------------------
