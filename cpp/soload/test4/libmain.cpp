//-------------------------------------------------------------------------------------------
/*! \file    libmain.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.25, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cmath>
#include <boost/filesystem.hpp>
//-------------------------------------------------------------------------------------------
// #include "libmain.h"
#include "test.h"
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
using namespace std;
using namespace boost;
using namespace boost::filesystem;

extern "C" void Print(void);

struct TInitializer
{
  TInitializer()
    {
      ExportFunction= "Print";
      cerr<<"libmain is loaded."<<endl;
      cerr<<"function "<<ExportFunction<<" is assigned to ExportFunction."<<endl;
    }
} TInitializer;

void Print(void)
{
  for(double x(-0.5);x<1.0;x+=0.3)
    cout<<"sin("<<x<<"): "<<sin(x)<<endl;

  path p("libmain.so");
  if (is_regular_file(p))        // is p a regular file?
    cout << p << " size is " << file_size(p) << '\n';
}


//-------------------------------------------------------------------------------------------
}  // end of loco_rabbits
//-------------------------------------------------------------------------------------------

