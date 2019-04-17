#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;  // 名前空間 Eigen を使う

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

int main(int,char**)
{
  // 表示用マクロ:
  #define printev(var)  \
    do{std::cout<<#var<<std::endl;var;}while(false)
  MatrixXd m(3,3);
  m<< 5,2,1,
      2,8,1,
      1,1,5;
  print(m);
  std::cout<<"###using PartialPivLU:"<<std::endl;
  PartialPivLU<MatrixXd> lu(m);
  int r;
  MatrixXd P,L,Ltmp,U;
  print(P=lu.permutationP());
  print(lu.matrixLU());
  print(r=std::max(m.rows(),m.cols()));
  print(U=lu.matrixLU().triangularView<Upper>());
  printev(Ltmp= MatrixXd::Identity(r,r));
  printev(Ltmp.block(0,0,m.rows(),m.cols()).triangularView<StrictlyLower>()= lu.matrixLU());
  print(Ltmp);
  print(L=Ltmp.block(0,0,P.cols(),U.rows()));
  print(P.inverse() * L * U);
  return 0;
}
