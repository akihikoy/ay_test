//-------------------------------------------------------------------------------------------
/*! \file    test1.cpp
    \brief   Simple yaml-cpp test
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.16, 2015

    Compile:
      g++ -g -Wall -o test1.out test1.cpp -Iyaml-cpp/include -Lyaml-cpp/build -lyaml-cpp
    Usage:
      ./test1.out
    Usage:
      ./test1.out /tmp/config.yaml
*/
//-------------------------------------------------------------------------------------------
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>  // getrusage, gettimeofday
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

typedef double t_real;
inline t_real GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<t_real>(time.tv_sec) + static_cast<t_real>(time.tv_usec)*1.0e-6l;
}

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
  std::string file_name("data1/config.yaml");
  if(argc>1)  file_name= argv[1];

  YAML::Node config = YAML::LoadFile(file_name);

  if (config["lastLogin"])
  {
    std::cout << "Last logged in: " << config["lastLogin"].as<std::string>() << std::endl;
  }

  const std::string username = config["username"].as<std::string>();
  const std::string password = config["password"].as<std::string>();
  std::cout << "Log in with: '" << username << "', '" << password << "'" << std::endl;
  config["lastLogin"] = GetCurrentTime();

  std::ofstream fout("/tmp/config.yaml");
  std::cout << "Wrote /tmp/config.yaml" << std::endl;
  fout << config;

  return 0;
}
//-------------------------------------------------------------------------------------------
