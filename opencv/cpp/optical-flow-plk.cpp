//-------------------------------------------------------------------------------------------
/*! \file    optical-flow-plk.cpp
    \brief   Test of calcOpticalFlowPyrLK.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb.08, 2017

src. opencv/samples/cpp/lkdemo.cpp

g++ -I -Wall -O2 optical-flow-plk.cpp -o optical-flow-plk.out -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui
*/
//-------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <ctype.h>
#include "cap_open.h"

using namespace cv;
using namespace std;

static void help()
{
  // print a welcome message, and the OpenCV version
  cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
          "Using OpenCV version " << CV_VERSION << endl;
  cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
  cout << "\nHot keys: \n"
          "\tESC - quit the program\n"
          "\tr - auto-initialize tracking\n"
          "\tc - delete all the points\n"
          "\tn - switch the \"night\" mode on/off\n"
          "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
  if( event == CV_EVENT_LBUTTONDOWN )
  {
    point = Point2f((float)x, (float)y);
    addRemovePt = true;
  }
}

int main( int argc, char** argv )
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

  help();

  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
  Size subPixWinSize(10,10), winSize(31,31);

  const int MAX_COUNT = 500;
  bool needToInit = false;
  bool nightMode = false;

  namedWindow( "LK Demo", 1 );
  setMouseCallback( "LK Demo", onMouse, 0 );

  Mat gray, prevGray, image, frame;
  vector<Point2f> points[2];

  for(;;)
  {
    if(!cap.Read(frame))
    {
      if(cap.WaitReopen()) continue;
      else break;
    }

    frame.copyTo(image);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    if( nightMode )
      image = Scalar::all(0);

    if( needToInit )
    {
      // automatic initialization
      goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
      cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
      addRemovePt = false;
    }
    else if( !points[0].empty() )
    {
      vector<uchar> status;
      vector<float> err;
      if(prevGray.empty())
        gray.copyTo(prevGray);
      calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                            3, termcrit, 0, 0.001);
      size_t i, k;
      for( i = k = 0; i < points[1].size(); i++ )
      {
        if( addRemovePt )
        {
          if( norm(point - points[1][i]) <= 5 )
          {
            addRemovePt = false;
            continue;
          }
        }

        if( !status[i] )
          continue;

        points[1][k++] = points[1][i];
        circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
      }
      points[1].resize(k);
    }

    if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
    {
      vector<Point2f> tmp;
      tmp.push_back(point);
      cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
      points[1].push_back(tmp[0]);
      addRemovePt = false;
    }

    needToInit = false;
    imshow("LK Demo", image);

    char c = (char)waitKey(10);
    if( c == 27 )
      break;
    switch( c )
    {
    case 'r':
      needToInit = true;
      break;
    case 'c':
      points[0].clear();
      points[1].clear();
      break;
    case 'n':
      nightMode = !nightMode;
      break;
    }

    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);
  }

  return 0;
}
//-------------------------------------------------------------------------------------------
