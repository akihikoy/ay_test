//-------------------------------------------------------------------------------------------
/*! \file    struct-init.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.04, 2013
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
// #include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

  struct TElementSize
  {
    int Body    ;
    int GeomBX  ;
    int GeomCA  ;
    int GeomCY  ;
    int GeomPL  ;
    int GeomSP  ;
    int GeomTM  ;
    int JointBA ;
    int JointH1 ;
    int JointSL ;
    int JointUN ;
    int JointH2 ;
    int JointFX ;
    TElementSize() :
        Body    (0),
        GeomBX  (0),
        GeomCA  (0),
        GeomCY  (0),
        GeomPL  (0),
        GeomSP  (0),
        GeomTM  (0),
        JointBA (0),
        JointH1 (0),
        JointSL (0),
        JointUN (0),
        JointH2 (0),
        JointFX (0)
      {}
  };

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
  TElementSize num;
  return 0;
}
//-------------------------------------------------------------------------------------------
