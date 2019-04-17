//-------------------------------------------------------------------------------------------
/*! \file    bar2.cpp
    \brief   certain c++ unit file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Nov.15, 2012
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
//-------------------------------------------------------------------------------------------
#include <foo/foo1.h>
#include <foo/foo2.h>
#include <bar/bar1.h>
#include <bar/bar2.h>
#include <octave/config.h>
#include <octave/octave.h>
#include <octave/Matrix.h>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
using namespace std;
// using namespace boost;


void BPrint2()
{
  ColumnVector x(2); x(0)=2.1; x(1)=-1.15;
  cout<<"Bar-2's print..."<<endl;
  cout<<"> "; FPrint1();
  cout<<"> "; FPrint2();
  cout<<"> "; BPrint1();
  cout<<"vec: "<<x.transpose()<<endl;
  cout<<"..."<<endl;
}

//-------------------------------------------------------------------------------------------
}  // end of loco_rabbits
//-------------------------------------------------------------------------------------------

