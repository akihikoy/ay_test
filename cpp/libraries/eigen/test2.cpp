// use compile option: -I/home/akihiko/prg/src/eigen/eigen-eigen-599fc4f7c736
#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

int main()
{
  MatrixXf X(100, 200);             // 100行200列の行列
  MatrixXf Y(200, 50);              // 200行50列の行列
  X(11,22) = 333;                   // 11行22列目に333を代入
  // ...色々処理
  MatrixXf A = X * Y;               // X*Yの結果をZに代入

  MatrixXf Z = MatrixXf::Zero(100, 50); // 零行列で初期化
  // ...色々処理
  MatrixXf B = 1.5 * X * Y + 3.0 * Z;
  // X*Yなどの中間要素をつくらず直接Bに計算結果が代入

  cout<<B<<endl;
  return 0;
}
