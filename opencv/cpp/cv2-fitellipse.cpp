// g++ -g -Wall -O2 -o cv2-fitellipse.out cv2-fitellipse.cpp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -I/usr/include/opencv4

// http://docs.opencv.org/master/de/dc7/fitellipse_8cpp-example.html#gsc.tab=0

/********************************************************************************
*
*
*  This program is demonstration for ellipse fitting. Program finds
*  contours and approximate it by ellipses.
*
*  Trackbar specify threshold parametr.
*
*  White lines is contours. Red lines is fitting ellipses.
*
*
*  Autor:  Denis Burenkov.
*
*
*
********************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
static void help()
{
    cout <<
        "\nThis program is demonstration for ellipse fitting. The program finds\n"
        "contours and approximate it by ellipses.\n"
        "Call:\n"
        "./fitellipse [image_name -- Default sample/edge.jpg]\n" << endl;
}
int sliderPos = 70;
Mat image;
void processImage(int, void*);
int main( int argc, char** argv )
{
    // cv::CommandLineParser parser(argc, argv,
        // "{help h||}{@image|sample/edge.jpg|}"
    // );
    // if (parser.has("help"))
    // {
        // help();
        // return 0;
    // }
    string filename = "sample/edge.jpg";
    // string filename = "sample/rtrace1.png";
    image = imread(filename, 0);
    if( image.empty() )
    {
        cout << "Couldn't open image " << filename << "\n";
        return 0;
    }
    imshow("source", image);
    namedWindow("result", 1);
    // Create toolbars. HighGUI use.
    createTrackbar( "threshold", "result", &sliderPos, 255, processImage );
    processImage(0, 0);
    // Wait for a key stroke; the same function arranges events processing
    waitKey();
    return 0;
}
// Define trackbar callback functon. This function find contours,
// draw it and approximate it by ellipses.
void processImage(int /*h*/, void*)
{
    vector<vector<Point> > contours;
    Mat bimage = image >= sliderPos;
    findContours(bimage, contours, RETR_LIST, CHAIN_APPROX_NONE);
    Mat cimage = Mat::zeros(bimage.size(), CV_8UC3);
    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t count = contours[i].size();
        if( count < 6 )
            continue;
        Mat pointsf;
        Mat(contours[i]).convertTo(pointsf, CV_32F);
// cerr<<"pointsf="<<pointsf<<std::endl;
        RotatedRect box = fitEllipse(pointsf);
        if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30
          || MIN(box.size.width, box.size.height) < 5 )
            continue;
        drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);
cerr<<"box="<<box.center<<", "<<box.size<<", "<<box.angle<<std::endl;
        ellipse(cimage, box, Scalar(0,0,255), 1, cv::LINE_AA);
        ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, cv::LINE_AA);
        Point2f vtx[4];
        box.points(vtx);
        for( int j = 0; j < 4; j++ )
            line(cimage, vtx[j], vtx[(j+1)%4], Scalar(0,255,0), 1, cv::LINE_AA);
    }
    imshow("result", cimage);
}
