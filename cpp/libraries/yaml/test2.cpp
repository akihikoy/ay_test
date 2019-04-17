//-------------------------------------------------------------------------------------------
/*! \file    test2.cpp
    \brief   Simple yaml-cpp test
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2015

    Compile:
      g++ -g -Wall -o test2.out test2.cpp -Iyaml-cpp/include -Lyaml-cpp/build -lyaml-cpp
    Usage:
      ./test2.out
      ./test2.out data2/b51.yaml
      ./test2.out /tmp/config.yaml
*/
//-------------------------------------------------------------------------------------------
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string file_name("data2/b50.yaml");
  if(argc>1)  file_name= argv[1];

  YAML::Node node_b= YAML::LoadFile(file_name);

  // node.IsDefined IsNull IsScalar IsSequence IsMap
  if(node_b.IsMap())
  {
    if(node_b["help"].IsDefined() && node_b["help"].IsScalar())
      std::cout<<"Help: "<<node_b["help"].as<std::string>()<<std::endl;
    if(node_b["ref_marker_id"].IsDefined() && node_b["ref_marker_id"].IsScalar())
      std::cout<<"ref_marker_id: "<<node_b["ref_marker_id"].as<int>()<<std::endl;
    if(node_b["ref_marker_pose"].IsDefined() && node_b["ref_marker_pose"].IsSequence())
    {
      std::cout<<"ref_marker_pose: [";
      for(int i(0),i_end(node_b["ref_marker_pose"].size()); i<i_end; ++i)
        std::cout<<" "<<node_b["ref_marker_pose"][i].as<double>();
      std::cout<<"]"<<std::endl;
      std::cout<<"ref_marker_pose(2): [";
      for(YAML::const_iterator itr(node_b["ref_marker_pose"].begin()),itr_end(node_b["ref_marker_pose"].end()); itr!=itr_end; ++itr)
        std::cout<<" "<<itr->as<double>();
      std::cout<<"]"<<std::endl;
    }
    if(node_b["grab_primitives"].IsDefined() && node_b["grab_primitives"].IsSequence())
    {
      YAML::Node gp0= node_b["grab_primitives"][0];
      if(gp0.IsDefined() && gp0.IsMap())
      {
        std::cout<<"grab_primitives[0]: {"<<std::endl;
        for(YAML::const_iterator itr(gp0.begin()),itr_end(gp0.end()); itr!=itr_end; ++itr)
          if(itr->second.IsScalar())
            std::cout<<" " <<itr->first.as<std::string>()<<": "<<itr->second.as<std::string>()<<std::endl;
          else
            std::cout<<" " <<itr->first.as<std::string>()<<": ..."<<std::endl;
        std::cout<<"}"<<std::endl;
      }
    }
  }

  std::ofstream fout("/tmp/container.yaml");
  std::cout << "Wrote /tmp/container.yaml" << std::endl;
  fout << node_b;

  return 0;
}
//-------------------------------------------------------------------------------------------
