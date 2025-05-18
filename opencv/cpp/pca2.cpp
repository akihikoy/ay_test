//-------------------------------------------------------------------------------------------
/*! \file    pca2.cpp
    \brief   PCA with OpenCV
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021

g++ -I -Wall pca2.cpp -o pca2.out -I/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
// using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

cv::Mat LoadData(const std::string &filename, int num_cols)
{
  std::ifstream ifs(filename.c_str());
  std::string line;
  cv::Mat res, col(1, num_cols, CV_64F);
  while(std::getline(ifs,line))
  {
    std::stringstream ss(line);
    for(int c(0);c<num_cols;++c)
      ss>>col.at<double>(c);
    if(res.empty())
      res= col;
    else
      cv::vconcat(res,col, res);
  }
  return res;
}
//-------------------------------------------------------------------------------------------

template<typename T>
void SaveData(const std::string &filename, const cv::Mat &data)
{
  std::ofstream ofs(filename.c_str());
  std::string delim;
  for(int r(0);r<data.rows;++r)
  {
    delim= "";
    for(int c(0);c<data.cols;++c)
    {
      ofs<<delim<<data.at<T>(r,c);
      delim= " ";
    }
    ofs<<std::endl;
  }
}
//-------------------------------------------------------------------------------------------

cv::Mat ReduceDimWithPCA(const cv::Mat &data)
{
  cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Vec3d mean(pca.mean.at<double>(0,0), pca.mean.at<double>(0,1), pca.mean.at<double>(0,2));
  cv::Vec3d evals;
  cv::Mat evecs(3,3,CV_64F);
  for(int i(0);i<3;++i)  // i-th eigenvector
  {
    evals[i]= pca.eigenvalues.at<double>(i);
    for(int r(0);r<3;++r) evecs.at<double>(r,i)= pca.eigenvectors.at<double>(i,r);
  }
  print(mean);
  print(evals);
  print(evecs);
  print(pca.mean);
  print(pca.eigenvalues);
  print(pca.eigenvectors);

  //Project points
  // print(pca.project(data));
  cv::Mat projected= pca.project(data);

  //Backproject points of reduced dimensions
  // print(projected(cv::Rect(0,0,2,projected.rows)));
  cv::Mat reduced;
  cv::hconcat(projected(cv::Rect(0,0,2,projected.rows)), cv::Mat::zeros(projected.rows,1,CV_64F), reduced);
  // print(reduced);
  cv::Mat backprojected= pca.backProject(reduced);
  // print(backprojected);

  return backprojected;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string filename= (argc>1) ? argv[1] : "sample/points1.dat";
  cv::Mat data(LoadData(filename, 3));
  cv::Mat backprojected= ReduceDimWithPCA(data);

  SaveData<double>("/tmp/res.dat", backprojected);
  std::cerr<<"#Plot by:"<<std::endl;
  std::cerr<<"qplot -x -3d "<<filename<<" /tmp/res.dat"<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
