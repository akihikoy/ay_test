//-------------------------------------------------------------------------------------------
/*! \file    boost-path.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jul.18, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <boost/filesystem.hpp>
#include <list>
#include <string>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

/*! \brief Get the executable's full-path */
std::string GetExecutablePath ()
{
  char buf[512];
  int siz= readlink("/proc/self/exe", buf, sizeof(buf)-1);
  if(siz>0)
  {
    buf[siz]='\0';
    return buf;
  }
  std::cerr<<"failed to get executable's path"<<std::endl;
  return "";
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  boost::filesystem::path exec(argv[0]);
  boost::filesystem::path cmpl(boost::filesystem::complete(argv[0]));
  boost::filesystem::path init(boost::filesystem::initial_path());
  boost::filesystem::path init_exec(boost::filesystem::initial_path()/exec);

  cout << "executable's path: " << exec << endl;
  cout << "complete path: " << cmpl << endl;
  cout << "initial path: " << init << endl;
  cout << "initial+executable path: " << init_exec << endl;

  cout << "GetExecutablePath(): " << GetExecutablePath() << endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
