//-------------------------------------------------------------------------------------------
/*! \file    demo1.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.19, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <boost/filesystem.hpp>
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
  if (argc < 2)
  {
    cout << "Usage: demo4.out path\n";
    return 1;
  }

  using namespace boost;
  using namespace boost::filesystem;
  boost::filesystem::path absolute_path;

  path p (argv[1]);   // p reads clearer than argv[1] in the following code

  if (exists(p))    // does p actually exist?
  {
    if (is_regular_file(p))        // is p a regular file?
      cout << p << " size is " << file_size(p) << '\n';

    else if (is_directory(p))      // is p a directory?
      cout << p << " is a directory\n";

    else
      cout << p << " exists, but is neither a regular file nor a directory\n";
  }
  else
    cout << p << " does not exist\n";

  return 0;
}
//-------------------------------------------------------------------------------------------
