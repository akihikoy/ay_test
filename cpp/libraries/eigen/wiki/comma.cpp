#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  {
    MatrixXd m(2,3);
    m<< 1.0,  2.0,  3.0,
        -1.0, -2.0, -3.0;
    print(m);
    Vector4d v;
    v<< 2.0, 1.0, 0.0, -1.0;
    print(v);
    v<< 2,2,2,3;
    print(v);
  }

//   {
//     MatrixXd m(2,3)
//       << 1.0,  2.0,  3.0,
//         -1.0, -2.0, -3.0;
//     print(m);
//   }

  {
    MatrixXd m(2,3);
    m<< 1.0,  2.0,  3.0,
        -1.0, -2.0, -3.0;
    print(m*(Vector3d()<<0,1,0).finished());
    print((MatrixXd(2,2)<<1.0,2.0, 3.0,4.0).finished()*2.0);
  }
  return 0;
}
