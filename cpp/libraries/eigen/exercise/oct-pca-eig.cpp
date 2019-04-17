#include <octave/config.h>
#include <octave/Matrix.h>
#include <iostream>
#include <fstream>
#include <vector>

// 共分散行列を計算
// x は各行がサンプルデータ
inline Matrix get_covariance (Matrix x)
{
  const int n(x.rows());
  x = x - ColumnVector(n,1.0) * x.sum(0).row(0) / static_cast<double>(n);
  return x.transpose() * x / static_cast<double>(n-1);
}

// greater 関数オブジェクト．ただし比較に ref(i) を使う
template <typename T>
struct __type_ref_greater
{
  const T &ref;
  __type_ref_greater (const T &r) : ref(r) {};
  bool operator() (int i, int j)
    {return ref(i)>ref(j);};
};

// x を大きい順に並びかえたインデックスを得る
void get_sorted_indices (const ColumnVector &x, std::vector<int> &idx)
{
  idx.resize (x.length());
  int i(0);
  for(std::vector<int>::iterator itr(idx.begin()); itr!=idx.end(); ++itr,++i) *itr=i;
  __type_ref_greater<ColumnVector> ref_greater(x);
  std::sort(idx.begin(), idx.end(), ref_greater);
}

using namespace std;

int main(int argc, char**argv)
{
  Matrix Data(1000,3); // データサイズは固定
  // データの読み込み (各行がサンプルデータ)
  ifstream ifs("d/sample1.dat");
  ifs >> Data;
  ifs.close();
  // 共分散行列の計算
  Matrix Cov (get_covariance(Data));
  cout<< "Cov= "<<endl<< Cov;
  // 固有値分解
  EIG eig(Cov);
  cout<<"eig.eigenvalues()= "<<endl<< eig.eigenvalues();
  cout<<"eig.eigenvectors()= "<<endl<< eig.eigenvectors();
  // 大きい順にソートしたインデックスを得る
  vector<int> idx;
  get_sorted_indices (real(eig.eigenvalues()), idx);
  // もとのデータを1次元に圧縮
  ofstream ofs("reduced1.dat");
  Matrix Tr (eig.eigenvectors().rows(),1); // 変換行列
  Tr.insert (real(eig.eigenvectors().column(idx[0])), 0,0);
  ofs<< Data * Tr * Tr.transpose();
  ofs.close();
  // もとのデータを2次元に圧縮
  ofs.open("reduced2.dat");
  Tr.resize (eig.eigenvectors().rows(),2);
  Tr.insert (real(eig.eigenvectors().column(idx[0])), 0,0);
  Tr.insert (real(eig.eigenvectors().column(idx[1])), 0,1);
  ofs<< Data * Tr * Tr.transpose();
  ofs.close();
  return 0;
}
