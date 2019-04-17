// implemented marker matching

#include <cv.h>
#include <highgui.h>

#include <cassert>
#include <iostream>

#include "edgeldetector.h"
#include "linesegment.h"
#include "buffer.h"

const char  * WINDOW_NAME  = "Marker Tracker";
//const CFIndex CASCADE_NAME_LEN = 2048;

struct TTetragon
{
  Vector2f c[4];
};

using namespace std;
cv::Mat  draw_image;

struct debugLine
{
  int x1, y1, x2, y2, r, g, b, t;
};

std::vector< debugLine > debugLines;

void debugDrawAll()
{
  cv::Point start, end;

  for( int i=0, s=debugLines.size(); i<s; i++ )
  {
    start.x = debugLines[i].x1;
    start.y = debugLines[i].y1;

    end.x = debugLines[i].x2;
    end.y = debugLines[i].y2;

    cv::line(draw_image, start, end, cv::Scalar(debugLines[i].r,debugLines[i].g,debugLines[i].b), debugLines[i].t);
  }

  debugLines.resize(0);
}

void debugDrawLine(int x1, int y1, int x2, int y2, int r, int g, int b, int t)
{
  debugLine newLine;

  newLine.x1 = x1; newLine.y1 = y1;
  newLine.x2 = x2; newLine.y2 = y2;
  newLine.r = r; newLine.g = g; newLine.b = b;
  newLine.t = t;

  debugLines.push_back(newLine);
}

void debugDrawPoint(int x1, int y1, int r, int g, int b, int t)
{
  debugDrawLine(x1-0, y1-1, x1+0, y1+1, r, g, b, t);
  debugDrawLine(x1-1, y1-0, x1+1, y1-0, r, g, b, t);
}

void RotCounterClockwise(cv::Mat &m)
{
  cv::transpose(m,m);
  cv::flip(m,m,0);
}
void RotClockwise(cv::Mat &m)
{
  cv::transpose(m,m);
  cv::flip(m,m,1);
}

// return 1:points are on a clockwise triangle, -1:counter-clockwise
// if the points are on a line, return -1 when p0 is center, 1 when p1 is center, 0 when p2 is center
int CheckClockwise(const Vector2f &p0, const Vector2f &p1, const Vector2f &p2)
{
  int dx1,dx2,dy1,dy2;
  dx1= p1.x-p0.x;
  dy1= p1.y-p0.y;
  dx2= p2.x-p0.x;
  dy2= p2.y-p0.y;

  if(dx1*dy2 > dy1*dx2 ) return 1;
  if(dx1*dy2 < dy1*dx2 ) return -1;
  if((dx1*dx2 <0) || (dy1*dy2 <0)) return -1;
  if((dx1*dx1 + dy1*dy1 < dx2*dx2 + dy2*dy2)) return 1;
  return 0;
}

cv::Mat LoadTemplate(const char *filename)
{
  cv::Mat img= cv::imread(filename,0);
  int tsize((img.cols<=img.rows) ? img.cols : img.rows);
  cv::resize (img, img, cv::Size(tsize,tsize), 0,0, CV_INTER_LINEAR);
  cv::threshold(img,img,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);
  return img;
}
void ARMarkerToClockwiseTetragon(const ARMarker &marker, TTetragon &t)
{
  if(CheckClockwise(marker.c1,marker.c2,marker.c3)==1)
  {
    t.c[0]= marker.c1;
    t.c[1]= marker.c2;
    t.c[2]= marker.c3;
    t.c[3]= marker.c4;
  }
  else
  {
    t.c[0]= marker.c1;
    t.c[1]= marker.c4;
    t.c[2]= marker.c3;
    t.c[3]= marker.c2;
  }
}
double CalcSimilarity(const TTetragon &marker, const cv::Mat &image, const cv::Mat &tmpl, int *direction=NULL)
{
cerr<<"s0"<<endl;
  if(CheckClockwise(marker.c[0],marker.c[1],marker.c[2])!=1 || CheckClockwise(marker.c[0],marker.c[2],marker.c[3])!=1)
    return 0.0;
  cv::Point2f src[4];
  for(int i(0);i<4;++i)
  {
    if(std::isinf(marker.c[i].x) || std::isinf(marker.c[i].y))
      return 0.0;
    src[i]= cv::Point2f(marker.c[i].x,marker.c[i].y);
  }
  cv::Point2f dst[4];
  dst[0]= cv::Point2f(0,0);
  dst[1]= cv::Point2f(tmpl.cols,0);
  dst[2]= cv::Point2f(tmpl.cols,tmpl.rows);
  dst[3]= cv::Point2f(0,tmpl.rows);

cerr<<"s1"<<endl;
for(int i(0);i<4;++i)cerr<<"("<<src[i].x<<","<<src[i].y<<")";
cerr<<endl;
  cv::Mat trans= cv::getPerspectiveTransform(src, dst);
cerr<<"s1.5"<<endl;
  cv::Mat detected;
  cv::warpPerspective(image, detected, trans, cv::Size(tmpl.cols,tmpl.rows));

cerr<<"s2"<<endl;
  cv::Mat tmp;
  cv::cvtColor(detected,tmp,CV_BGR2GRAY);
  detected= tmp;
cerr<<"s2.5"<<endl;
  cv::threshold(detected,tmp,0,1, cv::THRESH_BINARY|cv::THRESH_OTSU);
  detected= tmp;

cerr<<"s3"<<endl;
  cv::Mat matching;
  double similarity(0.0), s;
  for(int i(0);i<4;++i)
  {
    bitwise_xor(detected,tmpl,matching);
    s= 1-static_cast<double>(sum(matching)[0])/static_cast<double>(matching.cols*matching.rows);
    if(s>similarity)
    {
      similarity= s;
      if(direction)  *direction= i;
    }
    if(i<3)
      RotCounterClockwise(detected);
  }
cerr<<"s4"<<endl;
  return similarity;
}


int main (int argc, char * const argv[])
{

  bool writeVideo = false;

  // create all necessary instances
  cv::namedWindow(WINDOW_NAME,1);
  cv::namedWindow("marker",1);

  cv::VideoCapture  camera;
  cv::VideoWriter writer;
  bool isColor   = true;
  double fps     = 15.0;  // or 30
  int frameW  = 640; // 744 for firewire cameras
  int frameH  = 480; // 480 for firewire cameras

  camera.open(0);
  if(!camera.isOpened())
  {
    std::cerr<<"no camera!"<<std::endl;
    return -1;
  }
  if( writeVideo )
  {
    writer.open("out.mpeg",
                   // CV_FOURCC('P','I','M','1'),
                   CV_FOURCC('M','J','P','G'),
                   fps,cv::Size(frameW,frameH),isColor);
  }

  // template image
  cv::Mat template_image= LoadTemplate("marker3.png");
  cv::imshow("marker", template_image*255);

  // marker detection
  Buffer *buffer = new Buffer();
  EdgelDetector *edgelDetector = new EdgelDetector();

  edgelDetector->debugDrawLineSegments( false );
  edgelDetector->debugDrawPartialMergedLineSegments( false );
  edgelDetector->debugDrawMergedLineSegments( false );
  edgelDetector->debugDrawExtendedLineSegments( true );
  edgelDetector->debugDrawSectors( false );
  edgelDetector->debugDrawSectorGrids( false );
  edgelDetector->debugDrawEdges( false );
  edgelDetector->debugDrawCorners( false );

  edgelDetector->debugDrawMarkers( true );

  // get an initial frame and duplicate it for later work
  cv::Mat current_frame;
  camera >> current_frame;
  draw_image.create(cv::Size(current_frame.cols, current_frame.rows), CV_8UC3);

  // as long as there are images ...
  while(true)
  {
    // draw faces
    // cv::flip (current_frame, draw_image, 1);

cerr<<"p1"<<endl;
    camera >> current_frame;
    if(current_frame.cols*current_frame.rows==0)
      {cerr<<"capture failed"<<endl; continue;}
cerr<<"p2"<<endl;
    cv::resize (current_frame, draw_image, cv::Size(), 1.0,1.0, CV_INTER_LINEAR);


    // Perform a Gaussian blur
    //cv::Smooth( draw_image, draw_image, CV_GAUSSIAN, 3, 3 );

cerr<<"p3"<<endl;
    buffer->setBuffer((unsigned char *) draw_image.data, draw_image.cols, draw_image.rows);

cerr<<"p4"<<endl;
    edgelDetector->setBuffer(buffer);
    std::vector<ARMarker> markers = edgelDetector->findMarkers();

cerr<<"p5"<<endl;
    {
      double s;
      int d;
      TTetragon t;
      for(std::vector<ARMarker>::const_iterator itr(markers.begin()),last(markers.end());itr!=last;++itr)
      {
cerr<<"p51"<<endl;
        ARMarkerToClockwiseTetragon(*itr,t);
cerr<<"p52"<<endl;
        s= CalcSimilarity(t, current_frame, template_image, &d);
cerr<<"p52.5"<<endl;
        if(s > 0.8)
        {
// cerr<<"p521"<<endl;
          cout<<" "<<s<<","<<d<<"("<<t.c[0].x<<","<<t.c[0].y<<")";
          for(int i(0);i<4;++i)
            edgelDetector->drawLine(t.c[i].x, t.c[i].y, t.c[(i+1)%4].x, t.c[(i+1)%4].y, 0, (d==i?255:0), 255, (d==i?10:2));
// cerr<<"p522"<<endl;
        }
cerr<<"p53"<<endl;
      }
      // usleep(100000);
      cout<<endl;
    }

cerr<<"p6"<<endl;
    debugDrawAll();

    // just show the image
    cv::imshow(WINDOW_NAME, draw_image);

    if( writeVideo )
    {
      writer.write(draw_image);
    }

    // wait a tenth of a second for keypress and window drawing
    int key = cv::waitKey (10);
    if (key == 'q' || key == 'Q')
      break;

    switch( key ) {
      case '4':  edgelDetector->debugDrawLineSegments( !edgelDetector->drawLineSegments );
        break;
      case '5':  edgelDetector->debugDrawPartialMergedLineSegments( !edgelDetector->drawPartialMergedLineSegments );
        break;
      case '6':  edgelDetector->debugDrawMergedLineSegments( !edgelDetector->drawMergedLineSegments );
        break;
      case '7':  edgelDetector->debugDrawExtendedLineSegments( !edgelDetector->drawExtendedLineSegments );
        break;
      case '9':  edgelDetector->debugDrawMarkers( !edgelDetector->drawMarkers );
        break;
      case '1':  edgelDetector->debugDrawSectors( !edgelDetector->drawSectors );
        break;
      case '2':  edgelDetector->debugDrawSectorGrids( !edgelDetector->drawSectorGrids );
        break;
      case '3':  edgelDetector->debugDrawEdges( !edgelDetector->drawEdges );
        break;
      case '8':  edgelDetector->debugDrawCorners( !edgelDetector->drawCorners );
        break;
      default:
        break;
    }
cerr<<"p7"<<endl;
  }

  // be nice and return no error
  return 0;
}
