//-------------------------------------------------------------------------------------------
/*! \file    color_curve_detection1.cpp
    \brief   Detect color markers as polygon curves.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Jul.20, 2021

g++ -g -Wall -O2 -o color_curve_detection1.out color_curve_detection1.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "cv2-videoout2.h"
#include "rotate90n.h"
#include "cap_open.h"
#define LIBRARY
// #include "thinning/thinning.hpp"
// #include "thinning/thinning.cpp"
// #include "float_trackbar.cpp"
// #include "floyd_apsp.cpp"
#include "thinning_graph1.cpp"

//-------------------------------------------------------------------------------------------
namespace loco_rabbits
{
}
//-------------------------------------------------------------------------------------------
using namespace std;
// using namespace boost;
using namespace loco_rabbits;
//-------------------------------------------------------------------------------------------
// #define print(var) PrintContainer((var), #var"= ")
// #define print(var) std::cout<<#var"= "<<(var)<<std::endl
//-------------------------------------------------------------------------------------------

struct TCCDParameter
{
  float ResizeFactor;

  int ThreshH1;
  int ThreshS1;
  int ThreshV1;
  int ThreshH2;
  int ThreshS2;
  int ThreshV2;
  int NDilate1;
  int NErode1;

  // Kind of thinning method; ZHANGSUEN or GUOHALL
  std::string ThinningKind;

  // Epsilon parameter of the polygon approximation method.
  double ApproxEpsilon;

  TCCDParameter()
    {
      ResizeFactor= 0.5;
      ThreshH1= 84;
      ThreshS1= 73;
      ThreshV1= 10;
      ThreshH2= 119;
      ThreshS2= 255;
      ThreshV2= 255;
      NDilate1= 2;
      NErode1= 2;
      ApproxEpsilon= 3.0;
      ThinningKind= "GUOHALL";
    }
};

int main(int argc, char**argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  TCCDParameter params_;

  std::vector<std::string> ThinningKindList;
  ThinningKindList.push_back("ZHANGSUEN");
  ThinningKindList.push_back("GUOHALL");
  std::string win("Camera");
  cv::namedWindow(win,1);
  CreateTrackbar<float>("ResizeFactor", win, &params_.ResizeFactor, 0.1, 1.0, 0.1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshH1", win, &params_.ThreshH1, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshS1", win, &params_.ThreshS1, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshV1", win, &params_.ThreshV1, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshH2", win, &params_.ThreshH2, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshS2", win, &params_.ThreshS2, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("ThreshV2", win, &params_.ThreshV2, 0, 255, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("NDilate1:", win, &params_.NDilate1, 0, 10, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<int>("NErode1:", win, &params_.NErode1, 0, 10, 1, &TrackbarPrintOnTrack);
  CreateTrackbar<std::string>("ThinningKind:", win, &params_.ThinningKind, ThinningKindList, &TrackbarPrintOnTrack);
  CreateTrackbar<double>("ApproxEpsilon", win, &params_.ApproxEpsilon, 0.0, 10.0, 0.1, &TrackbarPrintOnTrack);

  cv::Mat frame;
  while(true)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    cv::Mat img_hsv, img_binary, img_bin_resized, img_thinning, img_thin_expanded;
    cv::Mat point_types;

    cv::cvtColor(frame, img_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(img_hsv,
                cv::Scalar(params_.ThreshH1, params_.ThreshS1, params_.ThreshV1),
                cv::Scalar(params_.ThreshH2, params_.ThreshS2, params_.ThreshV2), img_binary);
    cv::dilate(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1), params_.NDilate1);
    cv::erode(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1), params_.NErode1);

    if(params_.ResizeFactor!=1.0)
      cv::resize(img_binary, img_bin_resized, cv::Size(), params_.ResizeFactor, params_.ResizeFactor, cv::INTER_LINEAR);
    else
      img_bin_resized= img_binary;

    // Apply thinning to get a skeleton
    double t0= GetCurrentTime();
    if(params_.ThinningKind=="ZHANGSUEN")
      cv::ximgproc::thinning(img_bin_resized, img_thinning, cv::ximgproc::THINNING_ZHANGSUEN);
    else if(params_.ThinningKind=="GUOHALL")
      cv::ximgproc::thinning(img_bin_resized, img_thinning, cv::ximgproc::THINNING_GUOHALL);
    double t1= GetCurrentTime();

    // Apply the graph point detection and connection.
    std::vector<std::vector<cv::Point> > spine_polys;
    TThinningGraph graph= DetectNodesFromThinningImg(img_thinning, point_types);
    ConnectNodes(graph, point_types);
    ThinningGraphToSpinePolys(graph, point_types, spine_polys, params_.ApproxEpsilon);
    double t2= GetCurrentTime();

    std::cout<<"Computation times:"<<endl
      <<"  Thinning: "<<t1-t0<<" sec"<<endl
      <<"  Graph analysis: "<<t2-t1<<" sec"<<endl;

    // Visualize results
    if(params_.ResizeFactor!=1.0)
      cv::resize(img_thinning, img_thin_expanded, cv::Size(), 1.0/params_.ResizeFactor, 1.0/params_.ResizeFactor, cv::INTER_LINEAR);
    else
      img_thin_expanded= img_thinning;
    cv::Mat img_viz(frame.rows, frame.cols, CV_8UC3);
    cv::Mat in[] = {img_thin_expanded + 0.5*img_binary, img_thin_expanded, img_thin_expanded};
    int from_to[] = {0,0, 1,1, 2,2};
    cv::mixChannels(in, 3, &img_viz, 1, from_to, 3);
    img_viz= 0.5*frame + img_viz;
    DrawThinningGraph(img_viz, graph, spine_polys, 1.0/params_.ResizeFactor);
    cv::imshow(win, frame);
    cv::imshow("ThinningGraph", img_viz);

    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
