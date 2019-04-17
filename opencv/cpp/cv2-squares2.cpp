// #define OPENCV_LEGACY
#ifdef OPENCV_LEGACY
  #include <cv.h>
  #include <highgui.h>
#else
  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <opencv2/highgui/highgui.hpp>
#endif
#include <iostream>
#include <cmath>
#include <cstdio>

// based on: http://stackoverflow.com/questions/10533233/opencv-c-obj-c-advanced-square-detection
// compile: x++ -cv cv2-squares2.cpp -- -lopencv_core -lopencv_highgui -lopencv_imgproc

using namespace cv;
using namespace std;

Point2f computeIntersect(Vec2f line1, Vec2f line2);
vector<Point2f> lineToPointPair(Vec2f line);
bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta);


bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta)
{
  float theta1 = line1[1], theta2 = line2[1];

  if(theta1 < minTheta)
  {
    theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
  }

  if(theta2 < minTheta)
  {
    theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
  }

  return abs(theta1 - theta2) > minTheta;
}

// the long nasty wikipedia line-intersection equation...bleh...
Point2f computeIntersect(Vec2f line1, Vec2f line2)
{
  vector<Point2f> p1 = lineToPointPair(line1);
  vector<Point2f> p2 = lineToPointPair(line2);

  float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
  Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
             (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
            ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
             (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);

  return intersect;
}

vector<Point2f> lineToPointPair(Vec2f line)
{
  vector<Point2f> points;

  float r = line[0], t = line[1];
  double cos_t = cos(t), sin_t = sin(t);
  double x0 = r*cos_t, y0 = r*sin_t;
  double alpha = 1000;

  points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
  points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));

  return points;
}

// int main(int argc, char* argv[])
// {
  // Mat occludedSquare = imread("Square.jpg");

  // resize(occludedSquare, occludedSquare, Size(0, 0), 0.25, 0.25);

  // Mat occludedSquare8u;
  // cvtColor(occludedSquare, occludedSquare8u, CV_BGR2GRAY);

  // Mat thresh;
  // threshold(occludedSquare8u, thresh, 200.0, 255.0, THRESH_BINARY);

  // GaussianBlur(thresh, thresh, Size(7, 7), 2.0, 2.0);

  // Mat edges;
  // Canny(thresh, edges, 66.0, 133.0, 3);

  // vector<Vec2f> lines;
  // HoughLines( edges, lines, 1, CV_PI/180, 50, 0, 0 );

  // cout << "Detected " << lines.size() << " lines." << endl;

  // // compute the intersection from the lines detected...
  // vector<Point2f> intersections;
  // for( size_t i = 0; i < lines.size(); i++ )
  // {
    // for(size_t j = 0; j < lines.size(); j++)
    // {
      // Vec2f line1 = lines[i];
      // Vec2f line2 = lines[j];
      // if(acceptLinePair(line1, line2, CV_PI / 32))
      // {
        // Point2f intersection = computeIntersect(line1, line2);
        // intersections.push_back(intersection);
      // }
    // }

  // }

  // if(intersections.size() > 0)
  // {
    // vector<Point2f>::iterator i;
    // for(i = intersections.begin(); i != intersections.end(); ++i)
    // {
      // cout << "Intersection is " << i->x << ", " << i->y << endl;
      // circle(occludedSquare, *i, 1, Scalar(0, 255, 0), 3);
    // }
  // }

  // imshow("intersect", occludedSquare);
  // waitKey();

  // return 0;
// }

int main(int argc, char **argv)
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

  // cv::namedWindow(wndname,1);
  cv::Mat frame;
  for(;;)
  {
    cap >> frame; // get a new frame from camera


    //>>>
    Mat occludedSquare = frame;

    // resize(occludedSquare, occludedSquare, Size(0, 0), 0.25, 0.25);

    Mat occludedSquare8u;
    cvtColor(occludedSquare, occludedSquare8u, CV_BGR2GRAY);

    Mat thresh;
    threshold(occludedSquare8u, thresh, 200.0, 255.0, THRESH_BINARY);

    GaussianBlur(thresh, thresh, Size(7, 7), 2.0, 2.0);

    Mat edges;
    Canny(thresh, edges, 66.0, 133.0, 3);

    vector<Vec2f> lines;
    HoughLines( edges, lines, 1, CV_PI/180, 50, 0, 0 );

    cout << "Detected " << lines.size() << " lines." << endl;

    // compute the intersection from the lines detected...
    vector<Point2f> intersections;
    for( size_t i = 0; i < lines.size(); i++ )
    {
      for(size_t j = 0; j < lines.size(); j++)
      {
        Vec2f line1 = lines[i];
        Vec2f line2 = lines[j];
        if(acceptLinePair(line1, line2, CV_PI / 32))
        {
          Point2f intersection = computeIntersect(line1, line2);
          intersections.push_back(intersection);
        }
      }

    }

    if(intersections.size() > 0)
    {
      vector<Point2f>::iterator i;
      for(i = intersections.begin(); i != intersections.end(); ++i)
      {
        cout << "Intersection is " << i->x << ", " << i->y << endl;
        circle(occludedSquare, *i, 1, Scalar(0, 255, 0), 3);
      }
    }

    imshow("intersect", occludedSquare);
    //<<<


    // cv::imshow("camera", frame);
    int c(cv::waitKey(10));
    if(c=='\x1b'||c=='q') break;
    // usleep(10000);
  }
  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
