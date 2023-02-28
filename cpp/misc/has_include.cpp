//-------------------------------------------------------------------------------------------
/*! \file    has_include.cpp
    \brief   Test of __has_include preprocessor directive.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.28, 2023

g++ -Wall -O2 has_include.cpp -o has_include.out
*/
//-------------------------------------------------------------------------------------------
#if __has_include(<iostream>)
#  include <iostream>
#  define HAS_iostream "YES"
#else
#  define HAS_iostream "NO"
#endif

#if __has_include(<iostream_hoge>)
#  include <iostream_hoge>
#  define HAS_iostream_hoge "YES"
#else
#  define HAS_iostream_hoge "NO"
#endif
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  using namespace std;
  cout<<"Test of __has_include"<<std::endl;
  cout<<"HAS iostream: "<<HAS_iostream<<std::endl;
  cout<<"HAS iostream_hoge: "<<HAS_iostream_hoge<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
