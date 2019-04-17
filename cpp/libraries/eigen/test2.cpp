// use compile option: -I/home/akihiko/prg/src/eigen/eigen-eigen-599fc4f7c736
#include <Eigen/Core>
#include <iostream>
using namespace Eigen;
using namespace std;

int main()
{
  MatrixXf X(100, 200);             // 100��200��ι���
  MatrixXf Y(200, 50);              // 200��50��ι���
  X(11,22) = 333;                   // 11��22���ܤ�333������
  // ...��������
  MatrixXf A = X * Y;               // X*Y�η�̤�Z������

  MatrixXf Z = MatrixXf::Zero(100, 50); // �����ǽ����
  // ...��������
  MatrixXf B = 1.5 * X * Y + 3.0 * Z;
  // X*Y�ʤɤ�������Ǥ�Ĥ��餺ľ��B�˷׻���̤�����

  cout<<B<<endl;
  return 0;
}
