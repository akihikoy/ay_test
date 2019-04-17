#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う
using namespace std;

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  MatrixXd m(4,5);
  m<< 1,2,3,4,5,
      5,4,3,2,1,
      0,1,0,1,0,
      9,6,9,6,9;
  print(m);
  print(m.row(0));
  print(m.row(2));
  print(m.col(1));
  cout<<"m.block<3,3>(1,1)= "<<endl<<m.block<3,3>(1,1)<<endl;
  int rows(2),cols(3);
  print(rows);
  print(cols);
  print(m.block(0,2,rows,cols));
  print(m.leftCols(1));
  print(m.bottomRows(2));
  print(m.bottomRightCorner(2,2));

  return 0;
}
