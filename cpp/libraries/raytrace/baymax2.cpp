//-------------------------------------------------------------------------------------------
/*! \file    baymax2.cpp
    \brief   Ray tracing of baymax2 using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.04, 2015

    Compile:
    g++ -g -Wall -O3 -o baymax2.out baymax2.cpp libdoncross.a -lopencv_core -lopencv_highgui
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

  Intersection background_intersection;
  background_intersection.distanceSquared= -1.0;
  DepthScene scene(background_intersection);

  Spheroid* out_head= new Spheroid(4.0, 3.0, 4.0);
  out_head->Move(0.0, 0.0, 0.0);
  double a1(0.2);
  Spheroid* in_head= new Spheroid(4.0-a1, 3.0-a1, 4.0-a1);
  in_head->Move(0.0, 0.0, 0.0);
  SetIntersection *head= new SetDifference(Vector(0.0, 0.0, 0.0), out_head, in_head);

  double a2(0.4);
  Sphere* r_eye= new Sphere(Vector( 1.5, a2, 3.4), 0.4);
  Sphere* l_eye= new Sphere(Vector(-1.5, a2, 3.4), 0.4);
  Cuboid* bar= new Cuboid(1.5, 0.05, 0.5);
  bar->Move(0.0, a2, 3.6);

  // NOTE: use SetUnion to test
  SetIntersection *tmp_1= new SetDifference(Vector(0.0, 0.0, 0.0), head, r_eye);
  SetIntersection *tmp_2= new SetDifference(Vector(0.0, 0.0, 0.0), tmp_1, l_eye);
  SetIntersection *baymax= new SetDifference(Vector(0.0, 0.0, 0.0), tmp_2, bar);

  // baymax->SetFullMatte(Color(1.0, 1.0, 1.0));
  baymax->Move(0.0, 0.0, 20.0);
  // baymax->RotateX(180.0);
  baymax->RotateY(180.0);
  baymax->RotateZ(180.0);
  // baymax->RotateZ(11.0);
  // baymax->RotateX(5.0);
  // baymax->RotateY(10.0);

  scene.AddSolidObject(baymax);


  TCameraInfo xtion;  // Simulate the Xtion sensor
  xtion.Width= 400;
  xtion.Height= 400;
  xtion.Fx= 400;
  xtion.Fy= 400;
  xtion.Cx= 200;
  xtion.Cy= 200;

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
      double astep(5.0), cstep(0.5);
      switch(c)
      {
      case 65361/*LEFT*/:   baymax->RotateZ(-astep);   break;
      case 65363/*RIGHT*/:  baymax->RotateZ(+astep);   break;
      case 65362/*UP*/:     baymax->RotateX(-astep);   break;
      case 65364/*DOWN*/:   baymax->RotateX(+astep);   break;
      case 65360/*HOME*/:   baymax->RotateY(-astep);   break;
      case 65367/*END*/:    baymax->RotateY(+astep);   break;
      case 130897/*S+LEFT*/:   baymax->Translate(-cstep,0.0,0.0);   break;
      case 130899/*S+RIGHT*/:  baymax->Translate(+cstep,0.0,0.0);   break;
      case 130898/*S+UP*/:     baymax->Translate(0.0,-cstep,0.0);   break;
      case 130900/*S+DOWN*/:   baymax->Translate(0.0,+cstep,0.0);   break;
      case 130896/*S+HOME*/:   baymax->Translate(0.0,0.0,-cstep);   break;
      case 130903/*S+END*/:    baymax->Translate(0.0,0.0,+cstep);   break;
      }
      depth_img= cv::Scalar::all(0);
      normal_img= cv::Scalar::all(0);
      scene.Render2(xtion, &depth_img, &normal_img);
      cv::imshow("depth", (1.0-depth_img));
      cv::imshow("normal", normal_img);
      cv::waitKey(50);
    }
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
