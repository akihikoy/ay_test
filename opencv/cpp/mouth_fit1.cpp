//-------------------------------------------------------------------------------------------
/*! \file    mouth_fit1.cpp
    \brief   Fitting a mouth (a set of 3D points) by optimizing its pose from an edge image
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.27, 2016

g++ -g -Wall -O2 -o mouth_fit1.out mouth_fit1.cpp -I/usr/include/eigen3 -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
//-------------------------------------------------------------------------------------------
namespace trick
{

template <typename t_value>
inline t_value Sq(const t_value &val)
{
  return val*val;
}
//-------------------------------------------------------------------------------------------

/* Pose to Eigen::Affine3d.
    x: [0-2]: position x,y,z, [3-6]: orientation x,y,z,w. */
template <typename t_value>
inline Eigen::Affine3d XToEigMat(const t_value x[7])
{
  Eigen::Affine3d res;
  res= Eigen::Translation3d(x[0],x[1],x[2])
          * Eigen::Quaterniond(x[6], x[3],x[4],x[5]);
  return res;
}
//-------------------------------------------------------------------------------------------

/* Orientation to Eigen::Quaterniond.
    x: [0-3]: orientation x,y,z,w. */
template <typename t_value>
inline Eigen::Quaterniond QToEigMat(const t_value x[4])
{
  return Eigen::Quaterniond(x[3], x[0],x[1],x[2]);
}
//-------------------------------------------------------------------------------------------

/* Eigen::Affine3d to position.
    x: [0-2]: position x,y,z. */
template <typename t_value>
inline void EigMatToP(const Eigen::Affine3d T, t_value x[3])
{
  const Eigen::Vector3d p= T.translation();
  x[0]= p[0];
  x[1]= p[1];
  x[2]= p[2];
}
//-------------------------------------------------------------------------------------------

/* Eigen::Quaterniond to orientation.
    x: [0-3]: orientation x,y,z,w. */
template <typename t_value>
inline void EigMatToQ(const Eigen::Quaterniond q, t_value x[4])
{
  x[0]= q.x();
  x[1]= q.y();
  x[2]= q.z();
  x[3]= q.w();
}
/* Eigen::Affine3d to orientation.
    x: [0-3]: orientation x,y,z,w. */
template <typename t_value>
inline void EigMatToQ(const Eigen::Affine3d T, t_value x[4])
{
  EigMatToQ(T.rotation(), x);
}
//-------------------------------------------------------------------------------------------

/* Eigen::Affine3d to pose.
    x: [0-2]: position x,y,z, [3-6]: orientation x,y,z,w. */
template <typename t_value>
inline void EigMatToX(const Eigen::Affine3d T, t_value x[7])
{
  const Eigen::Vector3d p= T.translation();
  x[0]= p[0];
  x[1]= p[1];
  x[2]= p[2];
  Eigen::Quaterniond q(T.rotation());
  x[3]= q.x();
  x[4]= q.y();
  x[5]= q.z();
  x[6]= q.w();
}
//-------------------------------------------------------------------------------------------

/* Compute xout = x2 * x1
    where x1,x2,xout: [0-2]: position x,y,z, [3-6]: orientation x,y,z,w. */
template <typename t_value>
inline void TransformX(const t_value x2[7], const t_value x1[7], t_value xout[7])
{
  EigMatToX(XToEigMat(x2)*XToEigMat(x1), xout);
}
//-------------------------------------------------------------------------------------------

/* Compute xout = x2 * p1
    where p1,xout: [0-2]: position x,y,z,
      x2: [0-2]: position x,y,z, [3-6]: orientation x,y,z,w. */
template <typename t_value>
inline void TransformP(const t_value x2[7], const t_value p1[3], t_value xout[3])
{
  EigMatToP(XToEigMat(x2)*Eigen::Translation3d(p1[0],p1[1],p1[2]), xout);
}
//-------------------------------------------------------------------------------------------



// Project points 3D onto a rectified image.
void ProjectPointsToRectifiedImg(const cv::Mat &points3d, const cv::Mat &P, cv::Mat &points2d)
{
  cv::Mat P2;
  P.convertTo(P2,points3d.type());
  // cv::Mat points3dh, points2dh;
  // cv::convertPointsToHomogeneous(points3d, points3dh);
  // points2dh= points3dh*P2.t();
  cv::Mat points2dh= points3d*P2(cv::Range(0,3),cv::Range(0,3)).t();
  cv::Mat p3= P2.col(3).t();
  for(int r(0),rows(points2dh.rows);r<rows;++r)
    points2dh.row(r)+= p3;
  cv::convertPointsFromHomogeneous(points2dh, points2d);
}
//-------------------------------------------------------------------------------------------


}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void GenerateEdgePoints(cv::Mat &l_points3d, int N)
{
  double rad=0.05, height=0.06;
  double dth(2.0*M_PI/(double)N);
  l_points3d.create(N,3,CV_32F);
  for(int i(0);i<N;++i)
  {
    l_points3d.at<float>(i,0)= rad*std::cos((double)i*dth);
    l_points3d.at<float>(i,1)= height;
    l_points3d.at<float>(i,2)= rad*std::sin((double)i*dth);
  }
}
//-------------------------------------------------------------------------------------------

void TransformPoints(const cv::Mat &l_points3d, const double pose[7], cv::Mat &points3d)
{
  int N(l_points3d.rows);
  points3d.create(N,3,l_points3d.type());
  for(int i(0);i<N;++i)
  {
    double pd[3]= {(double)l_points3d.at<float>(i,0),(double)l_points3d.at<float>(i,1),(double)l_points3d.at<float>(i,2)};
    double p2d[3];
    TransformP(pose,pd,p2d);
    points3d.at<float>(i,0)= p2d[0];
    points3d.at<float>(i,1)= p2d[1];
    points3d.at<float>(i,2)= p2d[2];
  }
}
//-------------------------------------------------------------------------------------------

template <typename t_value>
void DrawPoints(cv::Mat &img, const cv::Mat &points, const cv::Scalar &col)
{
  for(int r(0),r_end(points.rows);r<r_end;++r)
  {
    int p[2]= {(int)points.at<t_value>(r,0), (int)points.at<t_value>(r,1)};
    // img.at<cv::Vec3b>(p[1],p[0])= cv::Vec3b(col[0],col[1],col[2]);
    cv::circle(img, cv::Point(p[0],p[1]), 2, col);
  }
}
//-------------------------------------------------------------------------------------------

template <typename t_value>
double EvaluateEdgePoints(const cv::Mat &edges, const cv::Mat &points2d)
{
  double sum(0);
  for(int r(0),r_end(points2d.rows);r<r_end;++r)
  {
    int p[2]= {(int)points2d.at<t_value>(r,0), (int)points2d.at<t_value>(r,1)};
    std::cerr<<" "<<(int)edges.at<unsigned char>(p[1],p[0]);
    sum+= edges.at<unsigned char>(p[1],p[0]);
  }
  std::cerr<<endl;
  sum/= (double)points2d.rows;
  return sum;
}
//-------------------------------------------------------------------------------------------

int main(int argc, char**argv)
{
  cv::VideoCapture cap(0); // open the default camera
  if(argc==2)
  {
    cap.release();
    cap.open(atoi(argv[1]));
  }
  if(!cap.isOpened())  // check if we succeeded
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  cv::namedWindow("camera",1);
  cv::namedWindow("edges",1);

  int N(10);  // Number of edge points
  cv::Mat l_points3d;
  GenerateEdgePoints(l_points3d, N);
  double pose[7]= {0.0,0.1,0.3, 0.0,0.0,0.0,1.0};

  double proj[]={
      225.2266400839457, 0, 303.1599922180176, 0,
      0, 225.2266400839457, 236.2535667419434, 0,
      0, 0, 1, 0};
  cv::Mat P(3,4,CV_64F,proj);

  cv::Mat frame;
  for(int f(0);;++f)
  {
    cap >> frame; // get a new frame from camera

    // pose[0]= 0.7*std::cos((double)f/40.0);
    // pose[1]= 0.5*std::sin((double)f/60.0);

    cv::Mat frame_gray, edges;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::blur(frame_gray, edges, cv::Size(3,3));
    cv::Canny(edges, edges, /*lowThreshold=*/70, 210, /*kernel_size=*/3);

    cv::Mat points3d, points2d;
    TransformPoints(l_points3d, pose, points3d);
    ProjectPointsToRectifiedImg(points3d, P, points2d);
    DrawPoints<float>(frame,points2d,cv::Scalar(255,255,0));

    cv::blur(edges, edges, cv::Size(3,3));
    double eval= EvaluateEdgePoints<float>(edges, points2d);
    std::cerr<<"eval: "<<eval<<endl;
    cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
    DrawPoints<float>(edges,points2d,cv::Scalar(255,255,0));

    cv::imshow("camera", frame);
    cv::imshow("edges", edges);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
