//-------------------------------------------------------------------------------------------
/*! \file    base64.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Dec.28, 2010
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
// #include <lora/stl_ext.h>
// #include <lora/string.h>
// #include <lora/sys.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
#include "base64-test.h"
#include <iostream>
#include <iterator>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
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
  // if (argc==1)  {cerr<<"usage: ./a.out strings"<<endl; return 1;}

  if (argc==2)
  {
    {
      std::ostreambuf_iterator<char> cout_itr(cout);
      TOutBase64Iterator <std::ostreambuf_iterator<char> >  base64encoder(&cout_itr);

      for(char *vp=argv[1]; *vp!='\0'; ++vp,++base64encoder)
        *base64encoder= *vp;
    }
    // cout<<endl;
  }
  else if (argc==1)
  {
    std::istreambuf_iterator<char> cin_itr(cin);
    //*dbg*/for(;cin_itr!=std::istreambuf_iterator<char>();++cin_itr)cerr<<"("<<int(*cin_itr)<<")";
    TInBase64Iterator <std::istreambuf_iterator<char> >  base64decoder(&cin_itr);

    for(; !base64decoder.IsEndOfStream(); ++base64decoder)
      cout<<*base64decoder;//<<"("<<int(*base64decoder)<<")";
    cout<<endl;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
