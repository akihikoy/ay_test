// g++ -I -Wall calcHist_hsv.cpp -o calcHist_hsv.out $(pkg-config --cflags --libs opencv4)

// src. opencv/samples/cpp/tutorial_code/Histograms_Matching/calcHist_Demo.cpp

/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include "cap_open.h"

using namespace std;
using namespace cv;

/**
 * @function main
 */
int main( int argc, char** argv )
{
  cv::VideoCapture cap;
  if(argc>=2)  cap= CapOpen(argv[1], /*width=*/0, /*height=*/0);
  else         cap= CapOpen("0", /*width=*/0, /*height=*/0);
  if(!cap.isOpened())  return -1;

  Mat src, hsv;
  namedWindow("camera", WINDOW_AUTOSIZE );
  namedWindow("calcHist Demo", WINDOW_AUTOSIZE );

  while(1)
  {
    cap >> src;

    cvtColor( src, hsv, COLOR_BGR2HSV );

    /// Separate the image in 3 places ( H, S, V )
    vector<Mat> hsv_planes;
    split( hsv, hsv_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat h_hist, s_hist, v_hist;

    /// Compute the histograms:
    calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
//     normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//     normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//     normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
h_hist*= 0.02;
s_hist*= 0.02;
v_hist*= 0.02;

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
                        Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(s_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(s_hist.at<float>(i)) ),
                        Scalar( 255, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(v_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(v_hist.at<float>(i)) ),
                        Scalar( 255, 255, 255), 2, 8, 0  );
    }

    /// Display
    imshow("camera", src );
    imshow("calcHist Demo", histImage );
    char c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;

    if (c == ' ') {
        static int save_count = 0;
        stringstream ss_img, ss_csv;

        ss_img << "image_" << save_count << ".png";
        ss_csv << "hist_" << save_count << ".csv";

        imwrite(ss_img.str(), src);

        ofstream ofs(ss_csv.str());
        if (ofs.is_open()) {
            ofs << "bin,h,s,v" << endl;
            for (int i = 0; i < histSize; i++) {
                ofs << i << ","
                    << h_hist.at<float>(i) << ","
                    << s_hist.at<float>(i) << ","
                    << v_hist.at<float>(i) << endl;
            }
            cout << "[Saved] " << ss_img.str() << ", " << ss_csv.str() << endl;
            save_count++;
        } else {
            cerr << "File open error." << endl;
        }
    }
  }

  return 0;
}
