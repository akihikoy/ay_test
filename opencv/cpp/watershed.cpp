// Test watershed algorithm.
// src: https://github.com/Itseez/opencv/blob/master/samples/cpp/watershed.cpp
// g++ -g -Wall -O2 -o watershed.out watershed.cpp -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_videoio
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>

// #include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <iostream>
#include "cap_open.h"

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
            "Usage:\n"
            "./watershed [camera idx]\n" << endl;


    cout << "Hot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tw or SPACE - run watershed segmentation algorithm\n"
        "\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
        "\t  (before that, roughly outline several markers on the image)\n";
}
Mat markerMask, img;
Point prevPt(-1, -1);

static void onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", img);
        imshow("markerMask", markerMask);
    }
}

int main( int argc, char** argv )
{
    TCapture cap;
    if(!cap.Open(((argc>1)?(argv[1]):"0"), /*width=*/((argc>2)?atoi(argv[2]):0), /*height=*/((argc>3)?atoi(argv[3]):0)))  return -1;

    help();
    namedWindow( "image", 1 );
    namedWindow( "watershed transform", 1 );
    setMouseCallback( "image", onMouse, 0 );


    Mat img0;
    cap >> img0;
    cvtColor(img0, markerMask , COLOR_BGR2GRAY);
    markerMask = Scalar::all(0);

    while(true)
    {
        Mat imgGray, tmp;
        if(!cap.Read(img0))
        {
          if(cap.WaitReopen()) continue;
          else break;
        }

        img0.copyTo(img);
        cvtColor(img, tmp, COLOR_BGR2GRAY);
        cvtColor(tmp, imgGray, COLOR_GRAY2BGR);
        imshow( "image", img );

        int c = waitKey();

        if( (char)c == 27 || c == 'q' )
            break;

        if( (char)c == 'r' )
        {
            markerMask = Scalar::all(0);
            img0.copyTo(img);
            imshow( "image", img );
        }

        if( (char)c == 'w' || (char)c == ' ' )
        {
            int i, j, compCount = 0;
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            Mat tmp2;
            markerMask.copyTo(tmp2);
            findContours(tmp2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);  // findContours modifies the source image;
            printf( "number of contours = %d\n", int(contours.size()) );

            if( contours.empty() )
                continue;
            Mat markers(markerMask.size(), CV_32S);
            markers = Scalar::all(0);
            int idx = 0;
            for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
                drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
            printf( "number of contour components = %d\n", compCount );
            imshow( "markers", markers*INT_MAX/compCount );

            if( compCount == 0 )
                continue;

            vector<Vec3b> colorTab;
            for( i = 0; i < compCount; i++ )
            {
                int b = theRNG().uniform(0, 255);
                int g = theRNG().uniform(0, 255);
                int r = theRNG().uniform(0, 255);

                colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            double t = (double)getTickCount();
            watershed( img0, markers );
            t = (double)getTickCount() - t;
            printf( "execution time = %gms\n", t*1000./getTickFrequency() );

            Mat wshed(markers.size(), CV_8UC3);

            // paint the watershed image
            for( i = 0; i < markers.rows; i++ )
                for( j = 0; j < markers.cols; j++ )
                {
                    int index = markers.at<int>(i,j);
                    if( index == -1 )
                        wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
                    else if( index <= 0 || index > compCount )
                        wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
                    else
                        wshed.at<Vec3b>(i,j) = colorTab[index - 1];
                }

            wshed = wshed*0.5 + imgGray*0.5;
            imshow( "watershed transform", wshed );
        }
    }

    return 0;
}

