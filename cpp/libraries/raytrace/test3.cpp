//-------------------------------------------------------------------------------------------
/*! \file    test3.cpp
    \brief   Ray tracing test 3 using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.03, 2015

    Compile:
    g++ -g -Wall -O3 -o test3.out test3.cpp libdoncross.a -lopencv_core -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include "depthscene.h"
#include "depthscene.cpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//-------------------------------------------------------------------------------------------
namespace Imager
{
}  // Imager
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  using namespace Imager;

  // Intersection background_intersection;
  // background_intersection.distanceSquared= -1.0;
  DepthScene scene/*(background_intersection)*/;

  Cylinder* cylinder_out = new Cylinder(0.04, 0.10);
  Cylinder* cylinder_in = new Cylinder(0.037, 0.10);
  cylinder_in->Move(0.0, 0.0, 0.01);

  SetIntersection *cylinder= new SetDifference(Vector(0.0, 0.0, -0.05), cylinder_out, cylinder_in);
  cylinder->Move(0.0461, 0.1142, 0.6272);
  cylinder->RotateX(90.0);
  cylinder->RotateY(0.0);
  cylinder->RotateZ(0.0);

  scene.AddSolidObject(cylinder);


  TCameraInfo xtion;  // Simulate the Xtion sensor
  xtion.Width= 640;
  xtion.Height= 480;
  xtion.Fx= 540.916992;
  xtion.Fy= 542.752869;
  xtion.Cx= 317.022348;
  xtion.Cy= 230.86987;

  cv::Mat depth_img, normal_img;
  scene.Render2(xtion, &depth_img, &normal_img);
  cv::namedWindow("depth",1);
  cv::namedWindow("normal",1);
  cv::imshow("depth", (1.0-depth_img));
  cv::imshow("normal", normal_img);
  while(true)
  {
    int c(cv::waitKey());
    cerr<<"Hit:"<<(char)c<<","<<c<<std::endl;
    if(c=='\x1b'||c=='q')  break;
    else if((65360<=c && c<=65364) || c==65367
            || (130896<=c && c<=130900) || c==130903)
    {
      double astep(5.0), cstep(0.01);
      switch(c)
      {
      case 65361/*LEFT*/:   cylinder->RotateZ(-astep);   break;
      case 65363/*RIGHT*/:  cylinder->RotateZ(+astep);   break;
      case 65362/*UP*/:     cylinder->RotateX(-astep);   break;
      case 65364/*DOWN*/:   cylinder->RotateX(+astep);   break;
      case 65360/*HOME*/:   cylinder->RotateY(-astep);   break;
      case 65367/*END*/:    cylinder->RotateY(+astep);   break;
      case 130897/*S+LEFT*/:   cylinder->Translate(-cstep,0.0,0.0);   break;
      case 130899/*S+RIGHT*/:  cylinder->Translate(+cstep,0.0,0.0);   break;
      case 130898/*S+UP*/:     cylinder->Translate(0.0,-cstep,0.0);   break;
      case 130900/*S+DOWN*/:   cylinder->Translate(0.0,+cstep,0.0);   break;
      case 130896/*S+HOME*/:   cylinder->Translate(0.0,0.0,-cstep);   break;
      case 130903/*S+END*/:    cylinder->Translate(0.0,0.0,+cstep);   break;
      }
      depth_img= cv::Scalar::all(0);
      normal_img= cv::Scalar::all(0);
      scene.Render2(xtion, &depth_img, &normal_img);
      cv::imshow("depth", (1.0-depth_img));
      cv::imshow("normal", normal_img);
      cv::waitKey(10);
    }
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
