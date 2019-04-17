#define NOT_MAIN
#include "file-io.cpp"

// 表示用マクロ:
#define print(var)  \
  std::cout<<#var"= "<<std::endl<<(var)<<std::endl

using namespace Eigen;
using namespace std;
int main(int argc,char **argv)
{
  if(argc!=4)
  {
    cerr<<"error!\nusage: "<<argv[0]<<" FILE1 FILE2 OP"<<endl;
    return -1;
  }
  MatrixXd m1,m2;
  LoadFromFile(m1,argv[1]);
  LoadFromFile(m2,argv[2]);
  cout<<"Matrix1(m1):\n"<<m1<<endl;
  cout<<"Matrix2(m2):\n"<<m2<<endl;

  switch(argv[3][0])
  {
  case '+': print(m1+m2); break;
  case '-': print(m1-m2); break;
  case 'x': print(m1*m2); break;
  default: cerr<<"invalid operator\nselect from +,-,x"<<endl; return -1;
  }
  return 0;
}
