//-------------------------------------------------------------------------------------------
/*! \file    mouth_fit3.cpp
    \brief   Fitting a mouth (a set of 3D points) by optimizing its pose from an edge image
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.27, 2016

g++ -g -Wall -O2 -o mouth_fit3.out mouth_fit3.cpp cma_es/cmaes.c cma_es/boundary_transformation.c -I/usr/include/eigen3 -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lm -lopencv_videoio
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include "cma_es/test03.h"
#include <boost/bind.hpp>
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



struct TCameraRectifier
{
  cv::Mat map1_, map2_;
  void Setup(const cv::Mat &K, const cv::Mat &D, const cv::Mat &R, const cv::Size &size_in, const double &alpha, const cv::Size &size_out);
  void Rectify(cv::Mat &frame);
};
//-------------------------------------------------------------------------------------------
void TCameraRectifier::Setup(const cv::Mat &K, const cv::Mat &D, const cv::Mat &R, const cv::Size &size_in, const double &alpha, const cv::Size &size_out)
{
  cv::Mat P= cv::getOptimalNewCameraMatrix(K, D, size_in, alpha, size_out);
  cv::initUndistortRectifyMap(K, D, R, P, size_out, CV_16SC2, map1_, map2_);
}
//-------------------------------------------------------------------------------------------
void TCameraRectifier::Rectify(cv::Mat &frame)
{
  cv::Mat framer;
  cv::remap(frame, framer, map1_, map2_, cv::INTER_LINEAR);
  frame= framer;
}
//-------------------------------------------------------------------------------------------

struct TStereoRectifier
{
  cv::Mat R1, R2, P1, P2, Q;
  cv::Mat map11_, map12_;
  cv::Mat map21_, map22_;
  void Setup(const cv::Mat &K1, const cv::Mat &D1, const cv::Mat &K2, const cv::Mat &D2, const cv::Mat &R, const cv::Mat &T, const cv::Size &size_in, const double &alpha, const cv::Size &size_out);
  void Rectify(cv::Mat &frame1, cv::Mat &frame2);
};
//-------------------------------------------------------------------------------------------
void TStereoRectifier::Setup(const cv::Mat &K1, const cv::Mat &D1, const cv::Mat &K2, const cv::Mat &D2, const cv::Mat &R, const cv::Mat &T, const cv::Size &size_in, const double &alpha, const cv::Size &size_out)
{
  cv::stereoRectify(K1, D1, K2, D2, size_in, R, T.t(),
      R1, R2, P1, P2, Q,
      /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/alpha,
      size_out/*, Rect* validPixROI1=0, Rect* validPixROI2=0*/);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, size_out, CV_16SC2, map11_, map12_);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, size_out, CV_16SC2, map21_, map22_);
}
//-------------------------------------------------------------------------------------------
void TStereoRectifier::Rectify(cv::Mat &frame1, cv::Mat &frame2)
{
  cv::Mat frame1r, frame2r;
  cv::remap(frame1, frame1r, map11_, map12_, cv::INTER_LINEAR);
  cv::remap(frame2, frame2r, map21_, map22_, cv::INTER_LINEAR);
  frame1= frame1r;
  frame2= frame2r;
}
//-------------------------------------------------------------------------------------------

}  // trick
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

void GenerateEdgePoints(cv::Mat &l_points3d, int N, const double &rad=0.041, const double &height=0.06)
{
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
void PointSeqToGradients(const cv::Mat &points2d, cv::Mat &grads2d)
{
  int N(points2d.rows);
  grads2d.create(N,2,points2d.type());
  if(N<=1)  return;
  for(int i(0);i<N;++i)
  {
    t_value p1[2]= {points2d.at<t_value>(i,0), points2d.at<t_value>(i,1)};
    t_value &gx(grads2d.at<t_value>(i,0)), &gy(grads2d.at<t_value>(i,1));
    if(i==0)
    {
      t_value p2[2]= {points2d.at<t_value>(i+1,0), points2d.at<t_value>(i+1,1)};
      gx= p2[0]-p1[0];
      gy= p2[1]-p1[1];
    }
    else if(i==N-1)
    {
      t_value p0[2]= {points2d.at<t_value>(i-1,0), points2d.at<t_value>(i-1,1)};
      gx= p1[0]-p0[0];
      gy= p1[1]-p0[1];
    }
    else
    {
      t_value p0[2]= {points2d.at<t_value>(i-1,0), points2d.at<t_value>(i-1,1)};
      t_value p2[2]= {points2d.at<t_value>(i+1,0), points2d.at<t_value>(i+1,1)};
      gx= p2[0]-p0[0];
      gy= p2[1]-p0[1];
    }
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
void DrawPoints(cv::Mat &img, const cv::Mat &points, const cv::Mat &grads)
{
  for(int r(0),r_end(points.rows);r<r_end;++r)
  {
    int p[2]= {(int)points.at<t_value>(r,0), (int)points.at<t_value>(r,1)};
    int g[2]= {(int)(std::fabs(grads.at<t_value>(r,0))), (int)(std::fabs(grads.at<t_value>(r,1)))};
    float lg(std::sqrt(g[0]*g[0]+g[1]*g[1]));
    cv::Scalar col(0.0, 255*g[1]/lg, 255*g[0]/lg);
    // img.at<cv::Vec3b>(p[1],p[0])= cv::Vec3b(col[0],col[1],col[2]);
    cv::circle(img, cv::Point(p[0],p[1]), 2, col);
  }
}
//-------------------------------------------------------------------------------------------

template <typename t_value>
double EvaluateEdgePoints(const cv::Mat &edges_x, const cv::Mat &edges_y, const cv::Mat &points2d, const cv::Mat &grads2d, bool &is_feasible)
{
  int rows(edges_x.rows), cols(edges_x.cols);
  int num(0);
  double sum(0);
  for(int r(0),r_end(points2d.rows);r<r_end;++r)
  {
    int p[2]= {(int)points2d.at<t_value>(r,0), (int)points2d.at<t_value>(r,1)};
    // NOTE: Since we applied convertScaleAbs to edges, we apply fabs to gradients
    t_value g[2]= {std::fabs(grads2d.at<t_value>(r,0)), std::fabs(grads2d.at<t_value>(r,1))};
    if(p[0]>=0 && p[0]<cols && p[1]>=0 && p[1]<rows)
    {
      // sum+= edges.at<unsigned char>(p[1],p[0]);
      // sum+= (edges.at<unsigned char>(p[1],p[0])>20 ? 1.0 : 0.0);
      // sum+= std::log(1+edges.at<unsigned char>(p[1],p[0]));
      // NOTE: signed short if convertScaleAbs is not used, unsigned char if convertScaleAbs is used
      // float ex(edges_x.at<signed short>(p[1],p[0])), ey(edges_y.at<signed short>(p[1],p[0]));
      float ex(edges_x.at<unsigned char>(p[1],p[0])), ey(edges_y.at<unsigned char>(p[1],p[0]));
      float le(std::sqrt(ex*ex+ey*ey)), lg(std::sqrt(g[0]*g[0]+g[1]*g[1]));
      if(le>50.0 && lg>1.0e-4)
      {
        // NOTE: ex and ey are flipped because horizontal edge detection (x) detects y-directional gradient
        float dot= (ey*g[0]+ex*g[1])/(le*lg);
        // sum+= dot*le;
        // sum+= dot+std::atan(le)/3.14;
        // sum+= 1.0;
        // sum+= le;
        // sum+= (le>100.0 && dot>0.8 ? 1.0 : 0.0);
        sum+= dot*std::log(le);
        // sum+= dot*std::atan(le);
        // sum+= dot;
        // std::cerr<<" | "<<g[0]<<" "<<g[1]<<" "<<ex<<" "<<ey<<" "<<" "<<lg<<" "<<le<<" "<<dot;
        ++num;
      }
    }
  }
  // std::cerr<<" * "<<num<<"/"<<points2d.rows<<endl;
  sum/= (double)points2d.rows;
  if((double)num/(double)points2d.rows<0.4)  is_feasible= false;
  else is_feasible= true;
  return sum;
}
//-------------------------------------------------------------------------------------------

// Parameterization of pose. dp: difference from a reference pose pref.
void ParamToPose(const double pref[7], const double dp[6], double pose[7])
{
  pose[0]= pref[0]+dp[0];
  pose[1]= pref[1]+dp[1];
  pose[2]= pref[2]+dp[2];
  Eigen::Vector3d axis(dp+3);
  double angle(axis.norm());
  if(angle>1.0e-6)  axis/= angle;
  Eigen::Quaterniond q= Eigen::AngleAxisd(angle,axis) * QToEigMat(pref+3);
  EigMatToQ(q, pose+3);
}
//-------------------------------------------------------------------------------------------

double FObjEdgePoints(const cv::Mat &edges_x, const cv::Mat &edges_y, const cv::Mat &l_points3d, const double *param, const double *pose0, const cv::Mat &P, bool &is_feasible)
{
  double pose[7];
  ParamToPose(pose0,param,pose);
  cv::Mat points3d, points2d, grads2d;
  TransformPoints(l_points3d, pose, points3d);
  ProjectPointsToRectifiedImg(points3d, P, points2d);
  PointSeqToGradients<float>(points2d, grads2d);
  // DrawPoints<float>(frame,points2d,cv::Scalar(255,255,0));

  return -EvaluateEdgePoints<float>(edges_x, edges_y, points2d, grads2d, is_feasible);
  // std::cerr<<"eval: "<<eval<<endl;
  // cv::cvtColor(edges, edges, CV_GRAY2BGR);
  // DrawPoints<float>(edges,points2d,cv::Scalar(255,255,0));
}
//-------------------------------------------------------------------------------------------

void DetectEdges(const cv::Mat &frame, cv::Mat &edges_x, cv::Mat &edges_y)
{
  cv::Mat frame_gray;
  cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
  cv::blur(frame_gray, frame_gray, cv::Size(3,3));
  int scale(1), delta(0), ddepth(CV_16S);
  // Gradient X
  //cv::Scharr(frame_gray, edges_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(frame_gray, edges_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
  // cv::convertScaleAbs(edges_x, edges_x);
  // Gradient Y
  //cv::Scharr(frame_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(frame_gray, edges_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
  // cv::convertScaleAbs(edges_y, edges_y);
// std::cerr<<"edges_x.type() "<<edges_x.type()<<" "<<CV_16S<<" "<<CV_8U<<std::endl;
  cv::convertScaleAbs(edges_x, edges_x);
  cv::convertScaleAbs(edges_y, edges_y);
  // cv::dilate(edges_x,edges_x,cv::Mat(),cv::Point(-1,-1), 2);
  // cv::dilate(edges_y,edges_y,cv::Mat(),cv::Point(-1,-1), 2);
// std::cerr<<"edges_x.type() "<<edges_x.type()<<" "<<CV_16S<<" "<<CV_8U<<std::endl;
  // int bw(9);
  // cv::blur(edges_x, edges_x, cv::Size(bw,bw));
  // cv::blur(edges_y, edges_y, cv::Size(bw,bw));
  int bw(31);
  cv::GaussianBlur(edges_x, edges_x, cv::Size(bw,bw), 0, 0, cv::BORDER_DEFAULT);
  cv::GaussianBlur(edges_y, edges_y, cv::Size(bw,bw), 0, 0, cv::BORDER_DEFAULT);
}
//-------------------------------------------------------------------------------------------

void FitEdgePoints(const cv::Mat &frame, const cv::Mat &l_points3d, double *pose, const cv::Mat &P)
{
  cv::Mat edges_x, edges_y;
  DetectEdges(frame, edges_x, edges_y);

  TCMAESParams params;
  params.lambda= 100;
  params.stopMaxFunEvals= 2000;
  params.stopTolFun= 1.0e-6;
  params.stopTolFunHist= -1.0;  // Disable
  // params.diagonalCov= 1.0;
  params.PrintLevel= 1;

  const int Dim(6);
  double bounds[2][Dim]= {
    {-0.1,-0.1,-0.1, -0.2,-0.2,-0.2},
    {+0.1,+0.1,+0.1, +0.2,+0.2,+0.2} };
  double x0[Dim]= {0.0,0.0,0.0, 0.0,0.0,0.0};
  // double sig0[Dim]= {0.2,0.2,0.2, 0.1,0.1,0.1};
  double sig0[Dim]= {0.02,0.02,0.02, 0.01,0.01,0.01};
  double xres[Dim];
  MinimizeF(boost::bind(&FObjEdgePoints, edges_x, edges_y, l_points3d, _1, pose, P, _2),
      x0, sig0, Dim, bounds[0], bounds[1], /*bound_len=*/Dim, xres, params);
  // double xres[Dim]= {pose[0],pose[1],pose[2]};

  std::cout<<"\nxres=";
  for(int d(0);d<Dim;++d)  std::cout<<" "<<xres[d];
  bool is_feasible;
  std::cout<<"; "<<FObjEdgePoints(edges_x,edges_y,l_points3d,xres,pose,P,is_feasible);
  std::cout<<std::endl;
  // for(int d(0);d<Dim;++d)  pose[d]= xres[d];
  ParamToPose(pose,xres,pose);

  // visualize
  cv::Mat points3d, points2d, grads2d;
  TransformPoints(l_points3d, pose, points3d);
  ProjectPointsToRectifiedImg(points3d, P, points2d);
  PointSeqToGradients<float>(points2d, grads2d);
  // cv::convertScaleAbs(edges_x, edges_x);
  // cv::convertScaleAbs(edges_y, edges_y);
  cv::Mat grads[3]= {0.0*edges_x, edges_x, edges_y}, disp;
  cv::merge(grads,3,disp);
  // cv::addWeighted(edges_x, 0.5, edges_y, 0.5, 0, disp);
  // cv::blur(disp, disp, cv::Size(9,9));
  // DrawPoints<float>(disp,points2d,cv::Scalar(255,255,0));
  DrawPoints<float>(disp,points2d,grads2d);
  disp= frame*0.25+disp;
  cv::imshow("edges", disp);
}
//-------------------------------------------------------------------------------------------


#ifndef NO_MAIN

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

  int N(20);  // Number of edge points
  cv::Mat l_points3d;
  GenerateEdgePoints(l_points3d, N);

  double cam[]={2.7000561666543712e+02, 0., 3.2129431094891305e+02, 0.,
       2.7000561666543712e+02, 2.4041057587530642e+02, 0., 0., 1.};
  double dist[]={-2.1709994562183432e-01, 3.8225723859672531e-02, 0., 0., 0.};
  double rect[]={1.,0.,0., 0.,1.,0., 0.,0.,1.};
  double proj[]={
      225.2266400839457, 0, 303.1599922180176, 0,
      0, 225.2266400839457, 236.2535667419434, 0,
      0, 0, 1, 0};
  cv::Mat P(3,4,CV_64F,proj), K(3,3,CV_64F,cam), D(1,5,CV_64F,dist), R(3,3,CV_64F,rect);

  TCameraRectifier rectifier;
  rectifier.Setup(K, D, R, cv::Size(640,480), 1.0, cv::Size(640,480));

      double pose[7]= {0.0,0.1,0.3, 0.0,0.0,0.0,1.0};
      EigMatToQ(Eigen::Quaterniond(Eigen::AngleAxisd(0.5,Eigen::Vector3d(1.0,0.0,0.0))), pose+3);
  cv::Mat frame, disp;
  bool capturing(true), running(true);
  for(int f(0);;++f)
  {
    if(running)
    {
      if(capturing)
      {
        cap >> frame; // get a new frame from camera
        rectifier.Rectify(frame);
      }

      // pose[0]= 0.7*std::cos((double)f/40.0);
      // pose[1]= 0.5*std::sin((double)f/60.0);

      FitEdgePoints(frame, l_points3d, pose, P);

      frame.copyTo(disp);
      cv::Mat points3d, points2d, grads2d;
      TransformPoints(l_points3d, pose, points3d);
      ProjectPointsToRectifiedImg(points3d, P, points2d);
      PointSeqToGradients<float>(points2d, grads2d);
      // DrawPoints<float>(disp,points2d,cv::Scalar(255,255,0));
      DrawPoints<float>(disp,points2d,grads2d);
    }
    else
    {
      usleep(200*1000);
    }

    cv::imshow("camera", disp);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    else if(c=='c')  capturing=!capturing;
    else if(c==' ')  running=!running;
    else if(c=='r')
    {
      double pose2[7]= {0.0,0.1,0.3, 0.0,0.0,0.0,1.0};
      for(int d(0);d<7;++d)  pose[d]= pose2[d];
      EigMatToQ(Eigen::Quaterniond(Eigen::AngleAxisd(0.5,Eigen::Vector3d(1.0,0.0,0.0))), pose+3);
    }
    // usleep(10000);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------

#endif // NO_MAIN
