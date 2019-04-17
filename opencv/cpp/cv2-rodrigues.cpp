//-------------------------------------------------------------------------------------------
/*! \file    cv2-rodrigues.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.10, 2012
*/
//-------------------------------------------------------------------------------------------
#include <cv.h>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{

namespace nconst
{
  const double Epsilon(1.0e-6);
}


template <typename t_elem>
inline cv::Mat_<t_elem> GetWedge (const  cv::Vec<t_elem,3> &w)
{
  cv::Mat_<t_elem> wedge(3,3);
  wedge(0,0)=0.0;    wedge(0,1)=-w(2);  wedge(0,2)=w(1);
  wedge(1,0)=w(2);   wedge(1,1)=0.0;    wedge(1,2)=-w(0);
  wedge(2,0)=-w(1);  wedge(2,1)=w(0);   wedge(2,2)=0.0;
  return wedge;
}

template <typename t_elem>
inline cv::Mat_<t_elem> Rodrigues (const cv::Vec<t_elem,3> &w)
{
  double th= norm(w);
  if(th<nconst::Epsilon)  return cv::Mat_<t_elem>::eye(3,3);
  cv::Mat_<t_elem> w_wedge(3,3);
  w_wedge= GetWedge(w *(1.0/th));
  return cv::Mat_<t_elem>::eye(3,3) + w_wedge * std::sin(th) + w_wedge * w_wedge * (1.0-std::cos(th));
}

template <typename t_elem>
inline cv::Vec<t_elem,3> InvRodrigues (const cv::Mat_<t_elem> &R)
{
  double alpha= (R(0,0)+R(1,1)+R(2,2) - 1.0) / 2.0;;

  if((alpha-1.0 < nconst::Epsilon) && (alpha-1.0 > -nconst::Epsilon))
    return cv::Vec<t_elem,3>(0.0,0.0,0.0);
  else
  {
    cv::Vec<t_elem,3> w;
    double th = std::acos(alpha);
    double tmp= 0.5 * th / std::sin(th);
    w[0] = tmp * (R(2,1) - R(1,2));
    w[1] = tmp * (R(0,2) - R(2,0));
    w[2] = tmp * (R(1,0) - R(0,1));
    return w;
  }
}

template <typename t_elem>
inline cv::Mat_<t_elem> AverageRotations (const cv::Mat_<t_elem> &R1, const cv::Mat_<t_elem> &R2, const t_elem &w2)
{
  cv::Vec<t_elem,3> w= InvRodrigues(cv::Mat_<double>(R2*R1.t()));
  return Rodrigues(w2*w)*R1;
}


}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<endl<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::Mat_<double> R1(Rodrigues(cv::Vec<double,3>(1,1,0))), R2(Rodrigues(cv::Vec<double,3>(0,1,1)));
  print(R1);
  print(R2);
  print(cv::Mat(InvRodrigues(cv::Mat_<double>(R2.t()*R1))));
  print(AverageRotations(R1,R2,0.0));
  print(AverageRotations(R1,R2,0.1));
  print(AverageRotations(R1,R2,0.5));
  print(AverageRotations(R1,R2,0.9));
  print(AverageRotations(R1,R2,1.0));
  return 0;
}
//-------------------------------------------------------------------------------------------
