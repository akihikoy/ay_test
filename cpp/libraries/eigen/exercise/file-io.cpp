#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>

template<typename t_elem>
bool LoadFromFile(Eigen::Matrix<t_elem,Eigen::Dynamic,Eigen::Dynamic> &m, const char *filename)
{
  std::ifstream ifs(filename);
  if(!ifs)  return false;
  int rows(0),cols(0);
  ifs >> rows >> cols;
  m.resize(rows,cols);
  t_elem elem;
  std::string line;
  int r(0);
  while(r<rows && std::getline(ifs,line))
  {
    if(line=="") continue;
    std::stringstream ss(line);
    int c(0);
    while(c<cols && ss>>elem)
    {
      m(r,c)= elem;
      ++c;
    }
    ++r;
  }
  return true;
}
template<typename t_elem>
bool SaveToFile(Eigen::Matrix<t_elem,Eigen::Dynamic,Eigen::Dynamic> &m, const char *filename)
{
  std::ofstream ofs(filename);
  if(!ofs)  return false;
  ofs<<m.rows()<<" "<<m.cols()<<std::endl;
  ofs<<m<<std::endl;
  return true;
}

#ifndef NOT_MAIN
using namespace Eigen;
using namespace std;
int main(int,char**)
{
  MatrixXd m1;
  LoadFromFile(m1,"mat1.dat");
  cout<<"loaded:\n"<<m1<<endl;
  MatrixXd m2(3,4);
  m2<<1,0,1,0,
      0,2,2,0,
      -1,2,-1,2;
  SaveToFile(m2,"out-mat1.dat");
  cout<<"saved to out-mat1.dat"<<endl;
  return 0;
}
#endif // NOT_MAIN
