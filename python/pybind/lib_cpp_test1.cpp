//-------------------------------------------------------------------------------------------
/*! \file    lib_cpp_test1.cpp
    \brief   Example library written in C++.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Aug.15, 2022
*/
//-------------------------------------------------------------------------------------------
#include "lib_cpp_test1.h"
#include <cassert>
//-------------------------------------------------------------------------------------------
namespace test
{
//-------------------------------------------------------------------------------------------


int Add(int x, int y)
{
  return x+y;
}
//-------------------------------------------------------------------------------------------

std::vector<int> VecConcatenate(const std::vector<int> &x, const std::vector<int> &y)
{
  std::vector<int> z(x.size()+y.size());
  std::vector<int>::iterator zitr(z.begin());
  for(std::vector<int>::const_iterator xitr(x.begin()),xend(x.end()); xitr!=xend; ++xitr,++zitr)
    *zitr= *xitr;
  for(std::vector<int>::const_iterator yitr(y.begin()),yend(y.end()); yitr!=yend; ++yitr,++zitr)
    *zitr= *yitr;
  return z;
}
//-------------------------------------------------------------------------------------------

std::vector<std::vector<int> > MatAdd(const std::vector<std::vector<int> > &x, const std::vector<std::vector<int> > &y)
{
  assert(x.size()==y.size());
  std::vector<std::vector<int> > z(x.size());
  std::vector<std::vector<int> >::iterator zitr(z.begin());
  for(std::vector<std::vector<int> >::const_iterator xitr(x.begin()),xend(x.end()),yitr(y.begin());
      xitr!=xend; ++xitr,++yitr,++zitr)
  {
    assert(xitr->size()==yitr->size());
    zitr->resize(xitr->size());
    std::vector<int>::iterator z2itr(zitr->begin());
    for(std::vector<int>::const_iterator x2itr(xitr->begin()),x2end(xitr->end()),y2itr(yitr->begin());
        x2itr!=x2end; ++x2itr,++y2itr,++z2itr)
      (*z2itr)= (*x2itr)+(*y2itr);
  }
  return z;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
// class TTest
//-------------------------------------------------------------------------------------------

TTest::TTest()
  : x_(0), y_(0)
{
}
//-------------------------------------------------------------------------------------------

TTest::TTest(int x, int y)
  : x_(x), y_(y)
{
}
//-------------------------------------------------------------------------------------------

int TTest::Sum() const
{
  return x_+y_;
}
//-------------------------------------------------------------------------------------------

std::vector<int> TTest::XY() const
{
  std::vector<int> xy(2);
  xy[0]= x_;
  xy[1]= y_;
  return xy;
}
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
}  // end of test
//-------------------------------------------------------------------------------------------

