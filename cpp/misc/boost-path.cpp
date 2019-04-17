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


//===========================================================================================
class TBinExecutor
//===========================================================================================
{
public:

  TBinExecutor()
    :
      included_list_(NULL),
      ignore_export_(false)
    {}


  void SetCurrentDir (const boost::filesystem::path &current_dir)  {current_dir_= current_dir;}
  void SetIncludedList (std::list<std::string> *included_list)  {included_list_= included_list;}
  void SetIgnoreExport (bool ignore_export)  {ignore_export_= ignore_export;}

protected:

  enum TExecutionMode {emNormal=0, emFunctionDef, emCompositeDef, emEdit, emSkipIf, emSkipElse};

  boost::filesystem::path              current_dir_;
  std::list<std::string>               *included_list_;
  bool                                 ignore_export_;


};
//-------------------------------------------------------------------------------------------



}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  if (argc < 2)
  {
    cout << "Usage: tut2 path\n";
    return 1;
  }

  boost::filesystem::path absolute_path;

  TBinExecutor hoge;

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
