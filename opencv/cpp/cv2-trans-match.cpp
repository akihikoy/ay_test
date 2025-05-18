//-------------------------------------------------------------------------------------------
/*! \file    cv2-trans-match.cpp
    \brief   certain c++ source file
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Aug.10, 2012
*/
//-------------------------------------------------------------------------------------------
#include <cv.h>
#include <highgui.h>
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
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::namedWindow("original",1);
  cv::namedWindow("trans",1);
  cv::namedWindow("template",1);
  cv::namedWindow("match",1);
  cv::Mat img1, img2, img3;

  img1= cv::imread("sample.jpg");
  cv::imshow("original",img1);

  img3= cv::imread("marker.jpg",0);
  int tsize((img3.cols<=img3.rows) ? img3.cols : img3.rows);
  cv::resize (img3, img3, cv::Size(tsize,tsize), 0,0, cv::INTER_LINEAR);
  cv::threshold(img3,img3,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);
  cv::imshow("template",img3*255);

  cv::Point2f src[4]= {cv::Point2f(280,72), cv::Point2f(334,70), cv::Point2f(384,136), cv::Point2f(326,138)};
  cv::Point2f dst[4];
  dst[0]= cv::Point2f(0,0);
  dst[1]= cv::Point2f(img3.cols,0);
  dst[2]= cv::Point2f(img3.cols,img3.rows);
  dst[3]= cv::Point2f(0,img3.rows);

  cv::Mat trans= cv::getPerspectiveTransform(src, dst);
  cv::warpPerspective(img1, img1, trans, cv::Size(img3.cols,img3.rows));

  cv::cvtColor(img1,img1,cv::COLOR_BGR2GRAY);
  cv::threshold(img1,img1,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);

  cv::imshow("trans",img1*255);

  for(int i(0);i<4;++i)
  {
    bitwise_xor(img1,img3,img2);
    cv::imshow("match",img2*255);
    cout<<"match ratio: "<<1-static_cast<double>(sum(img2)[0])/static_cast<double>(img2.cols*img2.rows)<<endl;
    cv::waitKey(0);
    if(i<3)
    {
      cv::transpose(img3,img3);
      cv::flip(img3,img3,1);
    }
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
