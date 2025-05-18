//-------------------------------------------------------------------------------------------
/*! \file    cv2-clone-vs-copyTo.cpp
    \brief   clone vs copyTo
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Mar.24, 2023

g++ -g -Wall -O2 -o cv2-clone-vs-copyTo.out cv2-clone-vs-copyTo.cpp -lopencv_core -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
#define print_op(op) std::cout<<#op<<std::endl;op
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  {
    float array[]= {1.0,2.0,3.0};
    cv::Mat m1(1,3,CV_32F, array);
    cv::Mat m1_clone, m1_copyTo, m1_shallow_copy;

    m1_clone= m1.clone();
    m1.copyTo(m1_copyTo);
    m1_shallow_copy= m1;

    print(m1);
    print(m1_clone);
    print(m1_copyTo);
    print(m1_shallow_copy);

    print_op(m1*= 10);

    print(m1);
    print(m1_clone);
    print(m1_copyTo);
    print(m1_shallow_copy);

    print_op(m1_clone*= 20);

    print(m1);
    print(m1_clone);
    print(m1_copyTo);
    print(m1_shallow_copy);

    print_op(m1_copyTo*= 30);

    print(m1);
    print(m1_clone);
    print(m1_copyTo);
    print(m1_shallow_copy);
  }
  std::cout<<"-----------------"<<std::endl;
  {
    float array[]= {1.0,2.0,3.0,10,10,10};
    cv::Mat m1(1,3,CV_32F, array);
    cv::Mat m1_shallow_copy;
    cv::Mat m2(1,3,CV_32F, array+3);

    m1_shallow_copy= m1;

    print(m1);
    print(m1_shallow_copy);
    print(m2);

    print_op(m1= m2.clone());

    print(m1);
    print(m1_shallow_copy);
  }
  std::cout<<"-----------------"<<std::endl;
  {
    float array[]= {1.0,2.0,3.0,10,10,10};
    cv::Mat m1(1,3,CV_32F, array);
    cv::Mat m1_shallow_copy;
    cv::Mat m2(1,3,CV_32F, array+3);

    m1_shallow_copy= m1;

    print(m1);
    print(m1_shallow_copy);
    print(m2);

    print_op(m2.copyTo(m1));

    print(m1);
    print(m1_shallow_copy);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
