//-------------------------------------------------------------------------------------------
/*! \file    mouth_fit4.cpp
    \brief   Fitting a mouth (a set of 3D points) by optimizing its pose from an edge image
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jun.27, 2016

g++ -g -Wall -O2 -o mouth_fit4.out mouth_fit4.cpp cma_es/cmaes.c cma_es/boundary_transformation.c -I/usr/include/eigen3 -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lm -lopencv_videoio
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
#ifdef NO_MAIN
#  include "mouth_fit3.cpp"
#else
#  define NO_MAIN
#  include "mouth_fit3.cpp"
#  undef NO_MAIN
#endif
//-------------------------------------------------------------------------------------------
namespace trick
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace trick;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TFObjEdgePointsParams2
{
  const cv::Mat &edges1_x;
  const cv::Mat &edges1_y;
  const cv::Mat &edges2_x;
  const cv::Mat &edges2_y;
  const cv::Mat &l_points3d;
  const double *pose0;
  const cv::Mat &P1;
  const cv::Mat &P2;
  TFObjEdgePointsParams2(const cv::Mat &v_edges1_x, const cv::Mat &v_edges1_y, const cv::Mat &v_edges2_x, const cv::Mat &v_edges2_y, const cv::Mat &v_l_points3d, const double *v_pose0, const cv::Mat &v_P1, const cv::Mat &v_P2)
    :
      edges1_x     (v_edges1_x   ),
      edges1_y     (v_edges1_y   ),
      edges2_x     (v_edges2_x   ),
      edges2_y     (v_edges2_y   ),
      l_points3d   (v_l_points3d ),
      pose0        (v_pose0      ),
      P1           (v_P1         ),
      P2           (v_P2         )
    {}
};

double FObjEdgePoints2(const TFObjEdgePointsParams2 &p, const double *param, bool &is_feasible)
{
  double pose[7];
  ParamToPose(p.pose0,param,pose);
  cv::Mat points3d, points2d1, grads2d1, points2d2, grads2d2;
  TransformPoints(p.l_points3d, pose, points3d);
  ProjectPointsToRectifiedImg(points3d, p.P1, points2d1);
  ProjectPointsToRectifiedImg(points3d, p.P2, points2d2);
  PointSeqToGradients<float>(points2d1, grads2d1);
  PointSeqToGradients<float>(points2d2, grads2d2);

  bool is_feasible1(false), is_feasible2(false);
  double e1= -EvaluateEdgePoints<float>(p.edges1_x, p.edges1_y, points2d1, grads2d1, is_feasible1);
  double e2= -EvaluateEdgePoints<float>(p.edges2_x, p.edges2_y, points2d2, grads2d2, is_feasible2);
  is_feasible= is_feasible1 && is_feasible2;
  return e1+e2;
}
//-------------------------------------------------------------------------------------------

void FitEdgePoints2(const cv::Mat &frame1, const cv::Mat &frame2, const cv::Mat &l_points3d, double *pose, const cv::Mat &P1, const cv::Mat &P2)
{
  cv::Mat edges1_x, edges1_y;
  cv::Mat edges2_x, edges2_y;
  DetectEdges(frame1, edges1_x, edges1_y);
  DetectEdges(frame2, edges2_x, edges2_y);

  TCMAESParams params;
  params.lambda= 100;
  params.stopMaxFunEvals= 2000;
  params.stopTolFun= 1.0e-6;
  params.stopTolFunHist= -1.0;  // Disable
  // params.diagonalCov= 1.0;
  params.PrintLevel= 1;

  TFObjEdgePointsParams2 fparams(edges1_x, edges1_y, edges2_x, edges2_y, l_points3d, pose, P1, P2);
  const int Dim(6);
  double bounds[2][Dim]= {
    {-0.1,-0.1,-0.1, -0.2,-0.2,-0.2},
    {+0.1,+0.1,+0.1, +0.2,+0.2,+0.2} };
  double x0[Dim]= {0.0,0.0,0.0, 0.0,0.0,0.0};
  // double sig0[Dim]= {0.2,0.2,0.2, 0.1,0.1,0.1};
  double sig0[Dim]= {0.02,0.02,0.02, 0.01,0.01,0.01};
  double xres[Dim];
  MinimizeF(boost::bind(&FObjEdgePoints2, fparams, _1, _2),
      x0, sig0, Dim, bounds[0], bounds[1], /*bound_len=*/Dim, xres, params);
  // double xres[Dim]= {pose[0],pose[1],pose[2]};

  std::cout<<"\nxres=";
  for(int d(0);d<Dim;++d)  std::cout<<" "<<xres[d];
  bool is_feasible;
  std::cout<<"; "<<FObjEdgePoints2(fparams,xres,is_feasible);
  std::cout<<std::endl;
  // for(int d(0);d<Dim;++d)  pose[d]= xres[d];
  ParamToPose(pose,xres,pose);
  std::cout<<"pose=";
  for(int d(0);d<7;++d)  std::cout<<" "<<pose[d];
  std::cout<<std::endl;

  // visualize
  cv::Mat points3d, points2d1, grads2d1, points2d2, grads2d2;
  TransformPoints(l_points3d, pose, points3d);
  ProjectPointsToRectifiedImg(points3d, P1, points2d1);
  ProjectPointsToRectifiedImg(points3d, P2, points2d2);
  PointSeqToGradients<float>(points2d1, grads2d1);
  PointSeqToGradients<float>(points2d2, grads2d2);
  cv::Mat grads1[3]= {0.0*edges1_x, edges1_x, edges1_y}, disp1;
  cv::merge(grads1,3,disp1);
  cv::Mat grads2[3]= {0.0*edges2_x, edges2_x, edges2_y}, disp2;
  cv::merge(grads2,3,disp2);
  DrawPoints<float>(disp1,points2d1,grads2d1);
  DrawPoints<float>(disp2,points2d2,grads2d2);
  disp1= frame1*0.25+disp1;
  disp2= frame2*0.25+disp2;
  cv::imshow("edges1", disp1);
  cv::imshow("edges2", disp2);
}
//-------------------------------------------------------------------------------------------

#ifndef NO_MAIN

int main(int argc, char**argv)
{
  cv::VideoCapture cap1(1),cap2(2);
  if(argc>2)
  {
    cap1.release();
    cap2.release();
    cap1.open(atoi(argv[1]));
    cap2.open(atoi(argv[2]));
  }
  if(!cap1.isOpened() || !cap2.isOpened())
  {
    std::cerr<<"can not open camera!"<<std::endl;
    return -1;
  }
  std::cerr<<"camera opened"<<std::endl;

  // cv::namedWindow("camera",1);
  cv::namedWindow("edges1",1);
  cv::namedWindow("edges2",1);

  int N(20);  // Number of edge points
  cv::Mat l_points3d;
  // GenerateEdgePoints(l_points3d, N, /*rad=*/0.041, /*height=*/0.06);
  GenerateEdgePoints(l_points3d, N, /*rad=*/0.05, /*height=*/0.14);
  // GenerateEdgePoints(l_points3d, N, /*rad=*/0.016, /*height=*/0.10);

  double cam1[]={2.7000561666543712e+02, 0., 3.2129431094891305e+02, 0.,
       2.7000561666543712e+02, 2.4041057587530642e+02, 0., 0., 1.};
  double dist1[]={-2.1709994562183432e-01, 3.8225723859672531e-02, 0., 0., 0.};
  double cam2[]={2.7000561666543712e+02, 0., 3.0228575091293698e+02, 0.,
       2.7000561666543712e+02, 2.2971641509663974e+02, 0., 0., 1.};
  double dist2[]={-1.9374466744485819e-01, 2.8092266079079675e-02, 0., 0., 0.};
  double rot[]={9.9963515063367447e-01, 1.0374291411559551e-02,
       -2.4938718798267576e-02, -9.9971574697470256e-03,
       9.9983449722542295e-01, 1.5199835542229445e-02,
       2.5092278894434918e-02, -1.4944973592943616e-02,
       9.9957342166755825e-01};
  double trans[]={-9.6065601314912027e-02, 9.5419386186126908e-04,
       -3.2455653422260805e-03};
  // double proj1[]={
      // 225.2266400839457, 0, 303.1599922180176, 0,
      // 0, 225.2266400839457, 236.2535667419434, 0,
      // 0, 0, 1, 0};
  // double proj2[]={
      // 225.2266400839457, 0, 303.1599922180176, -21.64994394559522,
      // 0, 225.2266400839457, 236.2535667419434, 0,
      // 0, 0, 1, 0};
  cv::Mat P1/*(3,4,CV_64F,proj1)*/, K1(3,3,CV_64F,cam1), D1(1,5,CV_64F,dist1);
  cv::Mat P2/*(3,4,CV_64F,proj2)*/, K2(3,3,CV_64F,cam2), D2(1,5,CV_64F,dist2);
  cv::Mat R(3,3,CV_64F,rot), T(1,3,CV_64F,trans);

  TStereoRectifier rectifier;
  rectifier.Setup(K1, D1, K2, D2, R, T, cv::Size(640,480), 0.0, cv::Size(640,480));
  P1= rectifier.P1;
  P2= rectifier.P2;

      double pose[7]= {0.0,0.1,0.3, 0.0,0.0,0.0,1.0};
      EigMatToQ(Eigen::Quaterniond(Eigen::AngleAxisd(0.5,Eigen::Vector3d(1.0,0.0,0.0))), pose+3);
  cv::Mat frame1, frame2, disp;
  bool capturing(true), running(true);
  for(int f(0);;++f)
  {
    if(running)
    {
      if(capturing)
      {
        cap1 >> frame1;
        cap2 >> frame2;
        rectifier.Rectify(frame1, frame2);
      }

      FitEdgePoints2(frame1, frame2, l_points3d, pose, P1, P2);
    }
    else
    {
      usleep(200*1000);
    }

    // cv::imshow("camera", disp);
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
