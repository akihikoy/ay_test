//-------------------------------------------------------------------------------------------
/*! \file    spec-for-struct.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.11, 2010
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
// #include <lora/rand.h>
// #include <lora/small_classes.h>
#include <lora/stl_ext.h>
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
#include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TTest
{
  template <typename T>
  struct F
    {
      void Print(const T &x)
        {
          cout<<"print: "<<x<<endl;
        }
    };
  template <typename T>
  void Print(const T &x)
    {
      F<T>().Print(x);
    }
};

template <>
struct TTest::F<int>
{
  void Print(const int &x)
  {
    cout<<"print int: "<<x<<endl;
  }
};

template <typename T>
struct TTest::F<list<T> >
{
  void Print(const list<T> &x)
  {
    cout<<"print list: ";
    print_container(x);
  }
};

/*!
I want to write a following code (partial specialization of a struct type T),
but it is impossible (in gcc 4.4).
Thus, I implement the later one.
\code
template <typename T>
struct TTest::F<struct T>
{
  void Print(const struct T &x)
    {
      cout<<"print struct: "<<endl;
      cout<<"  .X= "<<x.Entity.X<<endl;
      cout<<"  .Y= "<<x.Entity.Y<<endl;
    }
};
\endcode
*/

template <typename t_type>
struct TStruct
{
  typedef t_type T;
  T &Entity;
  TStruct(T &entity) : Entity(entity) {}
  // operator T&() {return Entity;}
};

template <typename T>
TStruct<T> Struct(T &entity) {return TStruct<T>(entity);}

template <typename T>
struct TTest::F<TStruct<T> >
{
  void Print(const TStruct<T> &x)
    {
      cout<<"print struct: "<<endl;
      cout<<"  .X= "<<x.Entity.X<<endl;
      cout<<"  .Y= "<<x.Entity.Y<<endl;
    }
};

struct TItem
{
  int X;
  double Y;
};

int main(int argc, char**argv)
{
  TTest test;
  list<double> tmp1; tmp1.push_back(1.2); tmp1.push_back(-5.5); tmp1.push_back(0.001);
  TItem tmp2= {-12, 4.55};
  test.Print(2.23);
  test.Print(-10);
  test.Print(tmp1);
  test.Print(Struct(tmp2));
  return 0;
}
//-------------------------------------------------------------------------------------------
