//-------------------------------------------------------------------------------------------
/*! \file    approx-nrand.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.27, 2011
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
//-------------------------------------------------------------------------------------------
template<typename T>
inline T urand()
{
  return static_cast<T>(std::rand())/static_cast<T>(RAND_MAX);
}
template<typename T>
inline T approx_nrand()
{
  T r(0.0l);
  for(int i(0);i<12;++i) r+=urand<T>();
  return r-6.0l;
}
template<typename T> inline T sq(const T&x) {return x*x;}

using namespace std;

int main(int argc, char**argv)
{
// #define RAND urand<double>
#define RAND approx_nrand<double>
#define N 10000.0
  double sum(0.0),sum2(0.0);
  for(int i(0);i<N;++i)
  {
    cout<<RAND()<<" "<<RAND()<<endl;
    sum+=RAND();
    sum2+=sq(RAND());
  }
  cerr<<"mean: "<<sum/N<<endl;
  cerr<<"var: "<<sum2/N-sq(sum/N)<<endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
