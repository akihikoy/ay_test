//-------------------------------------------------------------------------------------------
/*! \file    points_contour1.cpp
    \brief   Get a contour (convex hull) of 3D points projected onto a plane.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jan.19, 2021

g++ -I -Wall points_contour1.cpp -o points_contour1.out -I/usr/include/opencv2 -lopencv_core -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <cmath>
#include <sys/time.h>  // gettimeofday
//-------------------------------------------------------------------------------------------

inline double GetCurrentTime(void)
{
  struct timeval time;
  gettimeofday (&time, NULL);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)*1.0e-6;
  // return ros::Time::now().toSec();
}
//-------------------------------------------------------------------------------------------

void LoadData(const std::string &filename, std::list<cv::Point3d> &points)
{
  std::ifstream ifs(filename.c_str());
  std::string line;
  points.clear();
  while(std::getline(ifs,line))
  {
    std::stringstream ss(line);
    cv::Point3d pt;
    ss>>pt.x>>pt.y>>pt.z;
    points.push_back(pt);
  }
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


// Fit a cylinder parameter to an object specified by img_depth(mask,roi) on a plane specified by normal_pl and center_pl.
// NOTE: The direction cylinder==normal_pl.
void FitCylinder(const std::list<cv::Point3d> &points_pl, double &r_cyl, cv::Mat &center_cyl)
{
  center_cyl.create(1, 3, CV_64F);
  center_cyl.setTo(0.0);
  for(std::list<cv::Point3d>::const_iterator ip_pl(points_pl.begin()),ip_pl_end(points_pl.end()); ip_pl!=ip_pl_end; ++ip_pl)
  {
    center_cyl.at<double>(0,0)+= ip_pl->x;
    center_cyl.at<double>(0,1)+= ip_pl->y;
    center_cyl.at<double>(0,2)+= ip_pl->z;
  }
  center_cyl/= static_cast<double>(points_pl.size());
  r_cyl= 0.0;
  double r(0.0);
  for(std::list<cv::Point3d>::const_iterator ip_pl(points_pl.begin()),ip_pl_end(points_pl.end()); ip_pl!=ip_pl_end; ++ip_pl)
  {
    r= cv::norm(*ip_pl - cv::Point3d(center_cyl));
    if(r>r_cyl)  r_cyl= r;
  }
}
//-------------------------------------------------------------------------------------------

void ConvexHull(const std::list<cv::Point3d> &points, cv::Mat &hull_3d, int step=1)
{
  cv::Mat data(std::ceil(double(points.size())/double(step)), 3, CV_64F);
  int ir(0);
  for(std::list<cv::Point3d>::const_iterator ip(points.begin()),ip_end(points.end()); ip!=ip_end; ++ip,++ir)
  {
    if(ir%step==0)
    {
      data.at<double>(ir/step,0)= ip->x;
      data.at<double>(ir/step,1)= ip->y;
      data.at<double>(ir/step,2)= ip->z;
    }
  }

  double scale(10000);
  cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW);
  //Project points
  cv::Mat projected= pca.project(data), projected_s, projected_i, hull_pj_i, tmp[2];
  projected_s= scale*projected(cv::Rect(0,0,2,projected.rows));
  projected_s.convertTo(projected_i, CV_32S);
  cv::convexHull(projected_i, hull_pj_i, /*clockwise=*/true);
  cv::split(hull_pj_i, tmp);
  cv::hconcat(tmp[0], tmp[1], hull_pj_i);
  cv::Mat hull_pj, hull_pj_3d;
// std::cerr<<"DEBUG:projected_i:"<<projected_i<<std::endl;
// std::cerr<<"DEBUG:hull_pj_i:"<<hull_pj_i<<std::endl;
// std::cerr<<"DEBUG:hull_pj_i.size,channels:"<<hull_pj_i.size()<<" "<<hull_pj_i.channels()<<std::endl;
  hull_pj_i.convertTo(hull_pj, CV_64FC1);
// std::cerr<<"DEBUG:"<<CV_64F<<" "<<hull_pj_i.type()<<" "<<CV_64FC2<<" "<<hull_pj.type()<<" "<<cv::Mat::zeros(hull_pj.rows,1,CV_64F).type()<<" "<<projected_s.type()<<std::endl;
  cv::hconcat(hull_pj/scale, cv::Mat::zeros(hull_pj.rows,1,CV_64F), hull_pj_3d);
// projected_s= projected_s/scale;
// std::cerr<<"DEBUG:"<<CV_64F<<" "<<projected_s.type()<<std::endl;
// std::cerr<<"DEBUG:"<<CV_64F<<" "<<projected_s.type()<<std::endl;
// cv::hconcat(projected_s, cv::Mat::zeros(projected_s.rows,1,CV_64F), hull_pj_3d);
  hull_3d= pca.backProject(hull_pj_3d);
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  std::string filename= (argc>1) ? argv[1] : "sample/points1.dat";
  std::list<cv::Point3d> points;
  LoadData(filename, points);

  double r_cyl;
  cv::Mat center_cyl;
  FitCylinder(points, r_cyl, center_cyl);
  std::cout<<"r_cyl: "<<r_cyl<<std::endl;
  std::cout<<"center_cyl: "<<center_cyl<<std::endl;

  double t_start= GetCurrentTime();
  cv::Mat hull_3d;
  ConvexHull(points, hull_3d);
  std::cerr<<"Computation time: "<<GetCurrentTime()-t_start<<std::endl;

  SaveData<double>("/tmp/hull.dat", hull_3d);

  std::cerr<<"#Plot by:"<<std::endl;
  std::cerr<<"qplot -x -3d "<<filename<<" w d /tmp/hull.dat"<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
