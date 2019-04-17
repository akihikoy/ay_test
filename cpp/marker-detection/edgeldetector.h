/*
 * 
 * This file is part of a marker detection algorithm.
 *
 * Copyright (C) 2010 by Infi b.v.
 * http://www.infi.nl/blog/view/id/56/
 *  
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _h_edgeldetector_
#define _h_edgeldetector_

#include "buffer.h"
#include "vector2f.h"
#include <vector>

class Edgel;
class LineSegment;

class ARMarker 
{
public:
	
	void reconstructCorners();
	
	std::vector<LineSegment> chain;
	Vector2f c1, c2, c3, c4;
};

class EdgelDetector
{
public:
	EdgelDetector();
	
	int edgeKernel( unsigned char* offset, const int pitch );
	int edgeKernelX( int x, int y );
	int edgeKernelY( int x, int y );
	
	std::vector<Edgel> findEdgelsInRegion(const int left, const int top, const int width, const int height );
	std::vector<ARMarker> findMarkers();

	void scanLine(int offset, int step, int max, int width, int y);
	bool extendLine( Vector2f startpoint, const Vector2f slope, const Vector2f gradient, Vector2f& endpoint, const int maxlength );
	Vector2f edgeGradientIntensity(int x, int y);

	void setBuffer(Buffer* buffer);

	std::vector<LineSegment> findLineSegment(std::vector<Edgel> edgels);
	std::vector<LineSegment> mergeLineSegments(std::vector<LineSegment> linesegments, int max_iterations);
	void extendLineSegments( std::vector<LineSegment> &lineSegments );
	std::vector<LineSegment> findLinesWithCorners(std::vector<LineSegment> &linesegments);
	
	void findChainOfLines( LineSegment &startSegment, bool atStartPoint, std::vector<LineSegment> &linesegments, std::vector<LineSegment> &chain, int &length);
	
	// debug functions
	
	void debugDrawLineSegments( bool draw ) { drawLineSegments = draw; }
	void debugDrawPartialMergedLineSegments( bool draw ) { drawPartialMergedLineSegments = draw; }
	void debugDrawMergedLineSegments( bool draw ) { drawMergedLineSegments = draw; }
	void debugDrawExtendedLineSegments( bool draw ) { drawExtendedLineSegments = draw; }
	void debugDrawCorners( bool draw ) { drawCorners = draw; }
	void debugDrawMarkers( bool draw ) { drawMarkers = draw; }
	void debugDrawSectors( bool draw ) { drawSectors = draw; }
	void debugDrawEdges( bool draw ) { drawEdges = draw; }
	void debugDrawSectorGrids( bool draw ) { drawSectorGrids = draw; }
	
	void drawLine( int x1, int y1, int x2, int y2, int r, int g, int b, int t);
	void drawPoint( int x, int y, int r, int g, int b, int t);
	void drawArrow( int x1, int y1, int x2, int y2, float xn, float yn, int r, int g, int b, int t);
	
private:
	Buffer* buffer;	

public:
	bool drawLineSegments, drawPartialMergedLineSegments, drawMergedLineSegments, drawExtendedLineSegments, drawMarkers, drawSectors, drawSectorGrids,drawEdges,drawCorners;
};
#endif
