// re-written using opencv-2

#include <cv.h>
#include <highgui.h>

#include <cassert>
#include <iostream>

#include "edgeldetector.h"
#include "linesegment.h"
#include "buffer.h"

const char  * WINDOW_NAME  = "Face Tracker";
//const CFIndex CASCADE_NAME_LEN = 2048;


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

int main (int argc, char * const argv[])
{

  bool useCamera = true;
  bool writeVideo = false;

  // create all necessary instances
  cv::namedWindow(WINDOW_NAME,1);

  cv::VideoCapture  camera;
  cv::VideoWriter writer;
  bool isColor   = true;
  double fps     = 15.0;  // or 30
  int frameW  = 640; // 744 for firewire cameras
  int frameH  = 480; // 480 for firewire cameras

  if(useCamera)
  {
    camera.open(0);
    if(!camera.isOpened())
    {
      std::cerr<<"no camera!"<<std::endl;
      return -1;
    }
  }
  if( useCamera && writeVideo )
  {
    writer.open("out.mpeg",
                   // CV_FOURCC('P','I','M','1'),
                   CV_FOURCC('M','J','P','G'),
                   fps,cv::Size(frameW,frameH),isColor);
  }

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

  if( !useCamera )
  {
    draw_image= cv::imread("marker2.png");
    if(!draw_image.data)
    {
      printf("Could not load image file");
      return -1;
    }
    cv::imshow(WINDOW_NAME, draw_image);

    buffer->setBuffer((unsigned char *)draw_image.data, draw_image.cols, draw_image.rows);

    edgelDetector->setBuffer(buffer);
    std::vector<ARMarker> markers = edgelDetector->findMarkers();

    debugDrawAll();

    // Show the processed image
    cv::imshow(WINDOW_NAME, draw_image);

    cv::waitKey(0);
    return 0;

  }
  else
  {
    // get an initial frame and duplicate it for later work
    cv::Mat current_frame;
    camera >> current_frame;
    draw_image.create(cv::Size(current_frame.cols, current_frame.rows), CV_8UC3);

    // as long as there are images ...
    while(true)
    {
      // draw faces
      // cv::flip (current_frame, draw_image, 1);

      camera >> current_frame;
      cv::resize (current_frame, draw_image, cv::Size(), 1.0,1.0, CV_INTER_LINEAR);


      // Perform a Gaussian blur
      //cv::Smooth( draw_image, draw_image, CV_GAUSSIAN, 3, 3 );

      buffer->setBuffer((unsigned char *) draw_image.data, draw_image.cols, draw_image.rows);

      edgelDetector->setBuffer(buffer);
      std::vector<ARMarker> markers = edgelDetector->findMarkers();

      for(std::vector<ARMarker>::iterator itr(markers.begin()),last(markers.end());itr!=last;++itr)
        cout<<" ("<<itr->c1.x<<","<<itr->c1.y<<")";
      cout<<endl;

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
    }

  }

  // be nice and return no error
  return 0;
}
