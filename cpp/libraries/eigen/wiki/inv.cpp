#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う
using namespace std;

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(3,3);
  m<< 1,2,3,
      0,1,0,
      2,1,0;
  print(m);
  print(m.inverse());
  print(m.inverse() * m);

  return 0;
}
