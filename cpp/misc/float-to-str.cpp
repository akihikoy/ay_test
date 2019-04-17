//-------------------------------------------------------------------------------------------
/*! \file    float-to-str.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    May.20, 2010
*/
//-------------------------------------------------------------------------------------------
// #include <lora/util.h>
#include <lora/common.h>
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
#include <iomanip>
#include <sstream>
// #include <vector>
// #include <list>
#include <cfloat>
#include <boost/lexical_cast.hpp>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

inline const std::string FloatToStr (const float &val)
{
  // return boost::lexical_cast<std::string>(val);
  const float EXP_TH(1.0e+8f);
  std::stringstream ss;
  if(val>EXP_TH || val<-EXP_TH)  ss<< std::scientific;
  ss<< std::setprecision(FLT_DIG+3) << val;
  return ss.str();
}

inline const std::string FloatToStr (const double &val)
{
  // return boost::lexical_cast<std::string>(val);
  const double EXP_TH(1.0e+8);
  std::stringstream ss;
  if(val>EXP_TH || val<-EXP_TH)  ss<< std::scientific;
  ss<< std::setprecision(DBL_DIG+3) << val;
  return ss.str();
}

inline const std::string FloatToStr (const long double &val)
{
  // return boost::lexical_cast<std::string>(val);
  const long double EXP_TH(1.0e+8l);
  std::stringstream ss;
  if(val>EXP_TH || val<-EXP_TH)  ss<< std::scientific;
  ss<< std::setprecision(LDBL_DIG+3) << val;
  return ss.str();
}

}
//-------------------------------------------------------------------------------------------
using namespace std;
using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) print_container((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  // if(argc==1){LERROR("usage: "<<argv[0]<<" NUM");return 1;}
  // long double value= lexical_cast<long double>(argv[1]);
  // print(FloatToStr(value));
  // print(lexical_cast<int>(FloatToStr(value)));
  cout<<"error: ";
  /*long double*/{
    for(long double value=1.0e-1000l; value<1.0e+1000l; value*=1.2)
    // for(long double value=1.0l; value<1.0e+30l; value*=2.0l)
    // for(long double value=1.0l; value>1.0e-30l; value/=2.0l)
    {
      string str(FloatToStr(value));
      // print(str);
      long double retranslated= lexical_cast<long double>(str);
      if(value!=retranslated) cout<< " " << value-retranslated <<"("<<str<<")" <<flush;
      // else cout<< "-" <<flush;
    }
  }//*/
  /*double*-/{
    for(double value=1.0e-300; value<1.0e+300; value*=1.1)
    // for(double value=1.0; value<1.0e+30; value*=2.0)
    // for(double value=1.0; value>1.0e-30; value/=2.0)
    {
      string str(FloatToStr(value));
      // print(str);
      double retranslated= lexical_cast<double>(str);
      if(value!=retranslated) cout<< " " << value-retranslated <<"("<<str<<")" <<flush;
      // else cout<< "-" <<flush;
    }
  }//*/
  /*float*-/{
    for(float value=1.0e-30f; value<1.0e+30f; value*=1.001f)
    // for(float value=1.0f; value<1.0e+30f; value*=2.0f)
    // for(float value=1.0f; value>1.0e-30f; value/=2.0f)
    {
      string str(FloatToStr(value));
      // print(str);
      float retranslated= lexical_cast<float>(str);
      if(value!=retranslated) cout<< " " << value-retranslated <<"("<<str<<")" <<flush;
      // else cout<< "-" <<flush;
    }
  }//*/
  cout<<endl;
  print(FLT_DIG);
  print(DBL_DIG);
  print(LDBL_DIG);
  print(INT_MAX);
  return 0;
}
//-------------------------------------------------------------------------------------------
