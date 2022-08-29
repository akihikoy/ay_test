//-------------------------------------------------------------------------------------------
/*! \file    lib_cpp_test1.h
    \brief   Example library written in C++.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.15, 2022
*/
//-------------------------------------------------------------------------------------------
#ifndef lib_cpp_test1_h
#define lib_cpp_test1_h
//-------------------------------------------------------------------------------------------
#include <vector>
//-------------------------------------------------------------------------------------------
namespace test
{
//-------------------------------------------------------------------------------------------

int Add(int x, int y=1);

std::vector<int> VecConcatenate(const std::vector<int> &x, const std::vector<int> &y);

std::vector<std::vector<int> > MatAdd(const std::vector<std::vector<int> > &x, const std::vector<std::vector<int> > &y);

class TTest
{
public:
  TTest();
  TTest(int x, int y);
  int& X() {return x_;}
  const int& X() const {return x_;}
  int& Y() {return y_;}
  const int& Y() const {return y_;}
  int Sum() const;
  virtual std::vector<int> XY() const;
protected:
  int x_, y_;
};
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of test
//-------------------------------------------------------------------------------------------
#endif // lib_cpp_test1_h
//-------------------------------------------------------------------------------------------
