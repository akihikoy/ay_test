//-------------------------------------------------------------------------------------------
/*! \file    test1_3.cpp
    \brief   Ray tracing test 1-3 using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.02, 2015

    Compile:
    g++ -g -Wall -O3 -o test1_3.out test1_3.cpp libdoncross.a

    Remarkable implementations:
    imager.h
      class Scene
      SolidObjectList solidObjectList;
    main.cpp
      void CylinderTest()
    scene.cpp
      Color Scene::TraceRay
      int Scene::FindClosestIntersection
      void Scene::SaveImage
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include "doncross/raytrace/raytrace/imager.h"
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
  using namespace Imager;

  Scene scene(Color(0.0, 0.0, 0.0));

  Cylinder* cylinder_out = new Cylinder(2.0, 5.0);
  Cylinder* cylinder_in = new Cylinder(1.7, 5.0);
  cylinder_in->Move(0.0, 0.0, 0.4);

  SetIntersection *cylinder= new SetDifference(Vector(0.0, 0.0, 0.0), cylinder_out, cylinder_in);
  cylinder->SetFullMatte(Color(1.0, 0.5, 0.5));
  cylinder->Move(0.0, 0.0, -50.0);
  cylinder->RotateX(-60.0);
  cylinder->RotateY(-12.0);

  scene.AddSolidObject(cylinder);
  scene.AddLightSource(LightSource(Vector(+35.0, +50.0, +20.0), Color(1.0, 1.0, 1.0)));

  const char *filename = "/tmp/cylinder.png";
  scene.SaveImage(filename, 300, 300, 6.0, 2);
  std::cout << "Wrote " << filename << std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
