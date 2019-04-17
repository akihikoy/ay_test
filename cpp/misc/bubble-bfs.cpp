//-------------------------------------------------------------------------------------------
/*! \file    bubble-bfs.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Jun.21, 2010

    itplot -i 0.1 -s 'set logscale y; set y2tics' hehe.dat u 1, hehe.dat u 2 ax x1y2

    itplot -i 0.1 -s 'set xrange [-1.5:1.5];set yrange [-1.5:1.5]' hoge.dat w p pt 7 ps 1
    itplot -3d -i 0.1 -s 'set xrange [-1.5:1.5];set yrange [-1.5:1.5];set zrange [-1.5:1.5]' hoge.dat w p pt 7 ps 3
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
// #include <lora/common.h>
// #include <lora/math.h>
// #include <lora/file.h>
#include <lora/rand.h>
// #include <lora/stl_ext.h>
#include <lora/stl_math.h>
// #include <lora/string.h>
#include <lora/type_gen.h>
// #include <lora/octave.h>
// #include <lora/octave_str.h>
// #include <lora/ctrl_tools.h>
// #include <lora/ode.h>
// #include <lora/ode_ds.h>
// #include <lora/vector_wrapper.h>
#include <fstream>
#include <numeric>
// #include <string>
// #include <vector>
// #include <list>
// #include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
#include <lora/small_classes.h>
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
  Srand();
  TBubbleSet bubbles;

  int dim= 3;
  TBubbleSet::TRealVector scale(dim, 1.0l);
  GenAt(scale,1)=2.0l;
  TBubbleSet::TRealVector xmin(dim, -1.0l), xmax(dim, 1.0l);
  // GenAt(xmin,2)=(GenAt(xmax,2)=0.5l);
  GenAt(xmin,2)=-(GenAt(xmax,2)=0.4l);
  bubbles.SetMarginRatio(0.2l);
  bubbles.SetScale(scale);
  bubbles.SetCenterMin(xmin);
  bubbles.SetCenterMax(xmax);

  // TReal avr_len= GenAt(bubbles.CenterMax(),0)-GenAt(bubbles.CenterMin(),0);
  TBubbleSet::TRealVector diff=(xmax-xmin);
  TReal avr_len= accumulate(GenBegin(diff),GenEnd(diff),0.0l)/static_cast<TReal>(GenSize(diff));

  int N=25;
  TReal time_step= 0.1l;
  bubbles.GenerateRandomly(N, 0.5l*avr_len/real_pow(static_cast<TReal>(N),1.0l/static_cast<TReal>(GenSize(bubbles.CenterMin()))));
  TReal first_acc(bubbles.Step(time_step));
  TMovingAverageFilter<TReal>  acc_avr;
  acc_avr.Initialize(100,0.0l,first_acc);
  // int ni(0);
  do
  {
    cout<<(acc_avr(bubbles.Step(time_step)))<<" \t "<<bubbles.Radius()<<endl;
    ofstream ofs("hoge.dat");
    // ofstream ofs(("bbl/frame"+IntToStr(ni++,4)+".dat").c_str());
    bubbles.PrintRadiusCenters(ofs);
    // for(int i(0);i<bubbles.Size();++i){ofs<<GenPrint(bubbles.Center(i))<<endl;}
    usleep(10000);
  } while (acc_avr()>0.0002l*first_acc);

  return 0;
}
//-------------------------------------------------------------------------------------------
