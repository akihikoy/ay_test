//-------------------------------------------------------------------------------------------
/*! \file    baymax.cpp
    \brief   Ray tracing of baymax using Don Cross's library (doncross).
    \author  Akihiko Yamaguchi, akihiko-y@is.naist.jp / ay@akiyam.sakura.ne.jp
    \version 0.1
    \date    Feb.04, 2015

    Compile:
    g++ -g -Wall -O3 -o baymax.out baymax.cpp libdoncross.a
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include "doncross/raytrace/raytrace/imager.h"
#include "doncross/raytrace/raytrace/chessboard.h"
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

  // Scene scene(Color(0.0, 0.0, 0.0));
  Scene scene(Color(0.37, 0.37, 0.45, 7.0e-5));

  // Spheroid* baymax = new Spheroid(4.0, 3.0, 4.0);
  // Sphere* baymax= new Sphere(Vector(0.0, 0.0, 0.0), 4.0);

  // Spheroid* out_head = new Spheroid(4.0, 3.0, 4.0);
  // out_head->Move(0.0, 0.0, 0.0);
  // double a1(0.5);
  // Spheroid* in_head = new Spheroid(4.0-a1, 3.0-a1, 4.0-a1);
  // in_head->Move(0.0, 0.0, 0.0);
  // SetIntersection *baymax= new SetDifference(Vector(0.0, 0.0, 0.0), out_head, in_head);

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
  baymax->SetOpacity(0.2);
  baymax->SetMatteGlossBalance(0.9, Color(0.7,0.7,0.7), Color(0.8,1.0,0.7));
  baymax->Move(0.0, 0.0, -50.0);
  baymax->RotateZ(11.0);
  baymax->RotateX(5.0);
  baymax->RotateY(10.0);
  // baymax->RotateY(10.0);

  scene.AddSolidObject(baymax);




  // Create the chess board and add it to the scene.
  const double innerSize   = 20.00;
  const double xBorderSize = 1.00;
  const double yBorderSize = 1.00;
  const double thickness   = 0.25;
  const Color lightSquareColor = Color(0.75, 0.70, 0.10);
  const Color darkSquareColor  = Color(0.30, 0.30, 0.40);
  const Color borderColor      = Color(0.50, 0.30, 0.10);
  ChessBoard* board = new ChessBoard(
      innerSize,
      xBorderSize,
      yBorderSize,
      thickness,
      lightSquareColor,
      darkSquareColor,
      borderColor);
  board->Move(0.00, -5.3, -55.0);
  board->RotateZ(+11.0);
  board->RotateX(-62.0);
  scene.AddSolidObject(board);




  scene.AddLightSource(LightSource(Vector(-45.0, +25.0, +50.0), Color(0.4, 0.4, 0.1, 1.0)));
  scene.AddLightSource(LightSource(Vector( +5.0, +90.0, +40.0), Color(0.5, 0.5, 1.5, 1.0)));
  scene.AddLightSource(LightSource(Vector(-25.0, +30.0, +40.0), Color(0.3, 0.2, 0.1, 1.0)));

  // scene.AddLightSource(LightSource(Vector(0.0, 0.4, -50.0), Color(0.1, 0.1, 0.1, 0.5)));

  // scene.AddLightSource(LightSource(Vector(+35.0, +50.0, +20.0), Color(1.0, 1.0, 1.0)));
  // scene.AddLightSource(LightSource(Vector(-35.0, -50.0, +20.0), Color(1.0, 1.0, 1.0)));
  // scene.AddLightSource(LightSource(Vector(0.0, 0.0, -20.0), Color(1.0, 1.0, 1.0)));


  const char *filename = "/tmp/baymax.png";
  // scene.SaveImage(filename, 300, 300, 3.0, 1);
  scene.SaveImage(filename, 1000, 1000, 3.0, 1);
  std::cout << "Wrote " << filename << std::endl;

  return 0;
}
//-------------------------------------------------------------------------------------------
