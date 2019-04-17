//-------------------------------------------------------------------------------------------
/*! \file    relay-object.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.08, 2010
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
#include <iostream>
// #include <iomanip>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

class TTest
{
public:
  TTest() : entity_(NULL) {}
  TTest(TTest &x) : entity_(x.entity_) {x.entity_=NULL;}
  ~TTest()
    {
      Clear();
    }
  void Clear()
    {
      if(entity_)
      {
        std::cout<<"entity_("<<entity_<<","<<*entity_<<") is freed"<<std::endl;
        delete entity_;
      }
      entity_= NULL;
    }
  void Assign(const double &val)
    {
      if (entity_==NULL)
      {
        entity_= new double;
        std::cout<<"entity_("<<entity_<<","<<val<<") is allocated"<<std::endl;
      }
      *entity_= val;
    }
  const double& Entity() const {return *entity_;}

protected:
  TTest(const TTest &x);
  const TTest& operator= (const TTest &rhs);
  double *entity_;

};

/* WARNING: This function is useless because obj.entity_ is freed
     at the end of the function*/
TTest Generate (const double &init)
{
  TTest obj;
  obj.Assign(init);
  return obj;
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  TTest obj1(Generate(3.14));  // ERROR: ‘loco_rabbits::TTest::TTest(const loco_rabbits::TTest&)’ is protected
  TTest obj2;
  obj2= Generate(3.14);  // ERROR: ‘const loco_rabbits::TTest& loco_rabbits::TTest::operator=(const loco_rabbits::TTest&)’ is protected
  print(obj1.Entity());
  print(obj2.Entity());
  return 0;
}
//-------------------------------------------------------------------------------------------
