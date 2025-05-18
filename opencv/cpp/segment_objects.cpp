// g++ -I -Wall segment_objects.cpp -o segment_objects.out -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

// src. opencv/samples/cpp/segment_objects.cpp

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include "cap_open.h"

using namespace cv;
using namespace std;

static void help()
{
    printf("\n"
            "This program demonstrated a simple method of connected components clean up of background subtraction\n"
            "When the program starts, it begins learning the background.\n"
            "You can toggle background learning on and off by hitting the space bar.\n"
            "Call\n"
            "./segment_objects [video file, else it reads camera 0]\n\n");
}

static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 3;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp;

    dilate(mask, temp, Mat(), Point(-1,-1), niters);
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours( temp, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );

    dst = Mat::zeros(img.size(), CV_8UC3);

    if( contours.size() == 0 )
        return;

    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = 0;
    double maxArea = 0;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;
        }
    }
    Scalar color( 0, 0, 255 );
    drawContours( dst, contours, largestComp, color, cv::FILLED, 8, hierarchy );
}


int main(int argc, char** argv)
{
  TCapture cap;
  if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

    bool update_bg_model = true;

    help();

    Mat tmp_frame, bgmask, out_frame;

    cap >> tmp_frame;
    if(!tmp_frame.data)
    {
        printf("can not read data from the video source\n");
        return -1;
    }

    namedWindow("video", 1);
    namedWindow("segmented", 1);

    // BackgroundSubtractorMOG bgsubtractor;
    // bgsubtractor.set("noiseSigma", 3);
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgsubtractor=cv::createBackgroundSubtractorMOG2(/*int history=*/10, /*double varThreshold=*/5.0, /*bool detectShadows=*/true);

    for(;;)
    {
        // cap >> tmp_frame;
        // if( !tmp_frame.data )
        //     break;
        if(!cap.Read(tmp_frame))
        {
          if(cap.WaitReopen()) continue;
          else break;
        }
        bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
        refineSegments(tmp_frame, bgmask, out_frame);
        imshow("video", tmp_frame);
        imshow("segmented", out_frame);
        char keycode = waitKey(30);
        if( keycode == 27 )
            break;
        if( keycode == ' ' )
        {
            update_bg_model = !update_bg_model;
            printf("Learn background is in state = %d\n",update_bg_model);
        }
    }

    return 0;
}
