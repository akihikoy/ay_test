//-------------------------------------------------------------------------------------------
/*! \file    pca_normal.cpp
    \brief   PCA to get a normal vector from a depth image.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.13, 2021

g++ -I -Wall pca_normal.cpp -o pca_normal.out -I/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
//-------------------------------------------------------------------------------------------
namespace trick
{
inline unsigned Srand(void)
{
  unsigned seed ((unsigned)time(NULL));
  srand(seed);
  return seed;
}
inline double Rand (const double &max)
{
  return (max)*static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}
inline double Rand (const double &min, const double &max)
{
  return Rand(max - min) + min;
}
}
//-------------------------------------------------------------------------------------------
// using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
#define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------


cv::Mat GenPlane(const cv::Size &size, const double &h0, const cv::Vec3d &N, int n_zero=0)
{
  cv::Mat res(size, CV_64F);
  for(int r(0);r<size.height;++r)
    for(int c(0);c<size.width;++c)
      res.at<double>(r,c)= h0 - (N[0]*c+N[1]*r)/N[2] + Rand(-10,10);
  for(int i(0);i<n_zero;++i)
    res.at<double>(int(Rand(0,res.rows)),int(Rand(0,res.cols)))= 0.0;
  return res;
}
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

// Extract effective depth points and store them into Nx3 matrix.
template<typename t_img_depth>
cv::Mat DepthImgToPoints(const cv::Mat &img_patch, const double &d_scale=1.0, int step=1)
{
  // Extract effective depth points.
  int num_data(0);
  for(int r(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
      if(img_patch.at<t_img_depth>(r,c)>0)  ++num_data;
  cv::Mat points(num_data,3,CV_64F);
  for(int r(0),i(0);r<img_patch.rows;r+=step)
    for(int c(0);c<img_patch.cols;c+=step)
    {
      const double &d= img_patch.at<t_img_depth>(r,c);
      if(d>0)
      {
        points.at<double>(i,0)= c;
        points.at<double>(i,1)= r;
        points.at<double>(i,2)= d * d_scale;
        ++i;
      }
    }
  return points;
}
//-------------------------------------------------------------------------------------------

void GetNormal(const cv::Mat &points)
{
  // int num_data(0);
  // for(int r(0);r<img_patch.rows;r+=step)
  //   for(int c(0);c<img_patch.cols;c+=step)
  //     if(img_patch.at<double>(r,c)>0)  ++num_data;
  // if(num_data<3)  return /*cv::Mat()*/;
  // cv::Mat points(num_data,3,CV_64F);
  // for(int r(0),i(0);r<img_patch.rows;r+=step)
  //   for(int c(0);c<img_patch.cols;c+=step)
  //   {
  //     const double &h= img_patch.at<double>(r,c);
  //     if(h>0)
  //     {
  //       points.at<double>(i,0)= c;
  //       points.at<double>(i,1)= r;
  //       points.at<double>(i,2)= h;
  //       ++i;
  //     }
  //   }

  print(points.size());
  cv::PCA pca(points, cv::Mat(), CV_PCA_DATA_AS_ROW);
  cv::Mat normal= pca.eigenvectors.row(2);
  if(normal.at<double>(0,2)<0)  normal= -normal;

  print(pca.mean);
  print(pca.eigenvalues);
  print(pca.eigenvectors);
  print(normal);

  print(pca.eigenvalues.at<double>(2));  // thickness of the plane.

  cv::Mat viz_normal(2,3,CV_64F);
  for(int c(0);c<3;++c)  viz_normal.at<double>(0,c)= pca.mean.at<double>(0,c);
  for(int c(0);c<3;++c)  viz_normal.at<double>(1,c)= pca.mean.at<double>(0,c)+20*normal.at<double>(0,c);

  SaveData<double>("/tmp/plane.dat", points);
  SaveData<double>("/tmp/normal.dat", viz_normal);
  std::cerr<<"#Plot by:"<<std::endl;
  std::cerr<<"qplot -x -3d -s 'set view equal xyz' /tmp/plane.dat w p /tmp/normal.dat w l"<<std::endl;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  Srand();
  cv::Vec3d normal(Rand(-1,1),Rand(-1,1),Rand(0,1));
  cv::Mat plane= DepthImgToPoints<double>(GenPlane(cv::Size(30,30), 100, normal, /*n_zero=*/800));
  //cv::Mat plane= LoadData("sample/points1.dat", 3);
  //cv::Mat plane= LoadData("sample/points2.dat", 3);
  print(normal);
  cv::normalize(normal, normal);
  print(normal);
  // print(plane);
  GetNormal(plane);
  return 0;
}
//-------------------------------------------------------------------------------------------
