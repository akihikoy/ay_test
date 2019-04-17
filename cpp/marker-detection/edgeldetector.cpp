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

#include "edgeldetector.h"
#include <math.h>

#include <ctime>
#include <algorithm>
#include <iostream>

#include "edgel.h"
#include "linesegment.h"


#define THRESHOLD (16*16)
#define REGIONSIZE 40
#define EDGELSONLINE 5
#define RASTERSIZE 5
#define WHITETRESHHOLD 10
#define THICKNESS 2

EdgelDetector::EdgelDetector() {
	drawLineSegments = false;
	drawPartialMergedLineSegments = false;
	drawMergedLineSegments = false;
	drawExtendedLineSegments = false;
	drawCorners = false;
	drawMarkers = false;
	drawSectors = false;
	drawSectorGrids = false;
	drawEdges = false;
}


// debug functions

void debugDrawLine( int x1, int y1, int x2, int y2, int r, int g, int b, int t );
void debugDrawPoint( int x1, int y1, int r, int g, int b, int t );

void EdgelDetector::drawLine( int x1, int y1, int x2, int y2, int r, int g, int b, int t ) {
	debugDrawLine(x1, y1, x2, y2, r, g, b, t);
}

void EdgelDetector::drawPoint( int x, int y, int r, int g, int b, int t ) {
	debugDrawPoint(x, y, r, g, b, t);
}

void EdgelDetector::drawArrow( int x1, int y1, int x2, int y2, float xn, float yn, int r, int g, int b, int t ) {
	debugDrawLine(x1, y1, x2, y2, r, g, b, t);

	debugDrawLine( x2, y2, x2 + ( 5.0f* ( -xn + yn ) ), y2 + ( 5.0f* ( -yn - xn ) ), 0, 255, 0, t );
	debugDrawLine( x2, y2, x2 + ( 5.0f* ( -xn - yn ) ), y2 + ( 5.0f* ( -yn + xn ) ), 0, 255, 0, t );
}



// Marker

void ARMarker::reconstructCorners() {
	c1 = chain[0].getIntersection( chain[1] );
	c2 = chain[1].getIntersection( chain[2] );

	if( chain.size() == 4 ) {
		c3 = chain[2].getIntersection( chain[3] );
		c4 = chain[3].getIntersection( chain[0] );
	} else {
		c3 = chain[2].end.position;
		c4 = chain[0].start.position;
	}
}

// detection algorithm


int EdgelDetector::edgeKernel( unsigned char* offset, const int pitch ) {
	int ver = -3 * offset[ -2*pitch ];
	ver += -5 * offset[ -pitch ];
	ver += 5 * offset[ pitch ];
	ver += 3 * offset[ 2*pitch ];

	return abs( ver );
}

int EdgelDetector::edgeKernelX( int x, int y ) {
	int ver = -3 * buffer->getPixelColor( x, y-2, 0 );
	ver += -5 * buffer->getPixelColor( x, y-1, 0 );
	ver += 5 * buffer->getPixelColor( x, y+1, 0 );
	ver += 3 * buffer->getPixelColor( x, y+2, 0 );

	return abs( ver );
}


int EdgelDetector::edgeKernelY( int x, int y ) {
	int ver = -3 * buffer->getPixelColor( x-2, y, 0 );
	ver += -5 * buffer->getPixelColor( x-1, y, 0 );
	ver += 5 * buffer->getPixelColor( x+1, y, 0 );
	ver += 3 * buffer->getPixelColor( x+2, y, 0 );

	return abs( ver );
}

Vector2f EdgelDetector::edgeGradientIntensity(int x, int y) {
	int gx =  buffer->getPixelColor( x-1, y-1, 0 );
	gx += 2 * buffer->getPixelColor( x, y-1, 0 );
	gx += buffer->getPixelColor( x+1, y-1, 0 );
	gx -= buffer->getPixelColor( x-1, y+1, 0 );
	gx -= 2 * buffer->getPixelColor( x, y+1, 0 );
	gx -= buffer->getPixelColor( x+1, y+1, 0 );

	int gy = buffer->getPixelColor( x-1, y-1, 0 );
	gy += 2 * buffer->getPixelColor( x-1, y, 0 );
	gy += buffer->getPixelColor( x-1, y+1, 0 );
	gy -= buffer->getPixelColor( x+1, y-1, 0 );
	gy -= 2 * buffer->getPixelColor( x+1, y, 0 );
	gy -= buffer->getPixelColor( x+1, y+1, 0 );

	return Vector2f( float(gy), float(gx) ).get_normalized();
}

//vind alle markers in de buffer
std::vector<ARMarker>  EdgelDetector::findMarkers() {
	std::vector<LineSegment> mergedlinesegments;

	//maak regio's van REGIONSIZE*REGIONSIZE
	for( int y=2; y<buffer->getHeight()-3; y+=REGIONSIZE ) {
		for( int x=2; x<buffer->getWidth()-3; x+=REGIONSIZE) {
			//zoek edgels per regio
			std::vector<Edgel> edgels = findEdgelsInRegion(x, y, std::min( REGIONSIZE, buffer->getWidth()-x-3), std::min( REGIONSIZE, buffer->getHeight()-y-3) );

			std::vector<LineSegment> linesegments;

			//als er meer dan 5 edgels in de regio zitten, maak hier dan linesegments van
			if (edgels.size() > 5) {
				linesegments = findLineSegment( edgels );
			}

			// debug function
			if( drawLineSegments ) {
				for( int i=0, s = linesegments.size(); i<s; i++ ) {
					drawArrow( linesegments[i].start.position.x, linesegments[i].start.position.y, linesegments[i].end.position.x, linesegments[i].end.position.y,
							   linesegments[i].slope.x, linesegments[i].slope.y, 255, 0, 0, THICKNESS);
				}
			}

			//als er meer dan 1 linesegments gevonden zijn, merge ze dan
			if (linesegments.size() > 1) {
				linesegments = mergeLineSegments( linesegments, 50 );
			}

			// debug function
			if( drawPartialMergedLineSegments ) {
				for( int i=0, s = linesegments.size(); i<s; i++ ) {
					drawArrow( linesegments[i].start.position.x, linesegments[i].start.position.y, linesegments[i].end.position.x, linesegments[i].end.position.y,
							  linesegments[i].slope.x, linesegments[i].slope.y, 255, 0, 0, THICKNESS);				}
			}

			//en stop alle linesegments in een vector
			for (int i = 0; i < linesegments.size(); i++) {
				mergedlinesegments.push_back(linesegments.at(i));			}

		}
	}
	//merge de lines nog 1 keer over de gehele image
	mergedlinesegments = mergeLineSegments( mergedlinesegments, 50 );

	// debug function
	if( drawMergedLineSegments ) {
		for( int i=0, s = mergedlinesegments.size(); i<s; i++ ) {
			drawArrow( mergedlinesegments[i].start.position.x, mergedlinesegments[i].start.position.y, mergedlinesegments[i].end.position.x, mergedlinesegments[i].end.position.y,
					  mergedlinesegments[i].slope.x, mergedlinesegments[i].slope.y, 255, 0, 0, THICKNESS);		}
	}

	// extend linesegments

	extendLineSegments( mergedlinesegments );

	// debug function
	if( drawExtendedLineSegments ) {
		for( int i=0, s = mergedlinesegments.size(); i<s; i++ ) {
			drawArrow( mergedlinesegments[i].start.position.x, mergedlinesegments[i].start.position.y, mergedlinesegments[i].end.position.x, mergedlinesegments[i].end.position.y,
					  mergedlinesegments[i].slope.x, mergedlinesegments[i].slope.y, 255, 255, 0, THICKNESS );		}
	}

	std::vector<LineSegment> linesWithCorners = findLinesWithCorners( mergedlinesegments );

	if( drawCorners ) {
		for( int i=0, s = linesWithCorners.size(); i<s; i++ ) {
			if( linesWithCorners[i].start_corner ) { drawPoint( linesWithCorners[i].start.position.x, linesWithCorners[i].start.position.y, 255, 0, 255, THICKNESS); }
			if( linesWithCorners[i].end_corner ) { drawPoint( linesWithCorners[i].end.position.x, linesWithCorners[i].end.position.y, 255, 0, 255, THICKNESS); }


			drawArrow( linesWithCorners[i].start.position.x, linesWithCorners[i].start.position.y, linesWithCorners[i].end.position.x, linesWithCorners[i].end.position.y,
					  linesWithCorners[i].slope.x, linesWithCorners[i].slope.y, 0, 255, 0, THICKNESS);
		}
	}

	// detect markers
	std::vector<ARMarker> markers;

	if(linesWithCorners.size()) do {

		// pak een willekeurig segment, en probeer hier een chain mee te maken..
		LineSegment chainSegment = linesWithCorners[0];
		linesWithCorners[0] = linesWithCorners[ linesWithCorners.size() - 1 ];
		linesWithCorners.resize( linesWithCorners.size() - 1 );

		std::vector<LineSegment> chain;
		int length = 1;

		// kijk eerst of er schakels voor dit element moeten...
		findChainOfLines( chainSegment, true, linesWithCorners, chain, length);

		chain.push_back( chainSegment );

		// en misschien ook nog wel erna..
		if( length < 4 ) {
			findChainOfLines( chainSegment, false, linesWithCorners, chain, length);
		}

		if( length > 2 ) {
			ARMarker marker;

			marker.chain = chain;
			marker.reconstructCorners();

			markers.push_back( marker );
		}
	} while( linesWithCorners.size() );

	if( drawMarkers ) {
		for( int i=0, s=markers.size(); i<s; i++ ) {
			drawLine( markers[i].c1.x, markers[i].c1.y, markers[i].c2.x, markers[i].c2.y, 255, 0, 0, THICKNESS);
			drawLine( markers[i].c2.x, markers[i].c2.y, markers[i].c3.x, markers[i].c3.y, 255, 0, 0, THICKNESS);
			drawLine( markers[i].c3.x, markers[i].c3.y, markers[i].c4.x, markers[i].c4.y, 255, 0, 0, THICKNESS);
			drawLine( markers[i].c4.x, markers[i].c4.y, markers[i].c1.x, markers[i].c1.y, 255, 0, 0, THICKNESS);
		}
	}

	return markers;
}

void EdgelDetector::findChainOfLines( LineSegment &startSegment, bool atStartPoint, std::vector<LineSegment> &linesegments, std::vector<LineSegment> &chain, int& length) {
	const Vector2f startPoint = atStartPoint ? startSegment.start.position : startSegment.end.position;

	for( int i=0; i<linesegments.size(); i++ ) {
		// lijnen mogen niet parallel liggen
		if( startSegment.isOrientationCompatible( linesegments[i] ) ) {
			continue;
		}
		// eind en startpunt moeten dicht bij elkaar liggen...
		if( ( startPoint - ( atStartPoint ? linesegments[i].end.position :  linesegments[i].start.position ) ).get_squared_length() > 16.0f ) {
			continue;
		}
		// en de orientatie moet natuurlijk goed zijn, dus tegen de klok mee rond een zwart vierkantje
		if( ( atStartPoint &&
		      ( startSegment.slope.x * linesegments[i].slope.y - startSegment.slope.y * linesegments[i].slope.x <= 0 ) ) ||
		    ( !atStartPoint &&
			  ( startSegment.slope.x * linesegments[i].slope.y - startSegment.slope.y * linesegments[i].slope.x >= 0 ) ) ) {
				continue;
		}

		// het lijkt te mooi om waar te zijn, maar we hebben er 1 gevonden :)
		// haal dus dit segment er uit en kijk of de ketting langer te maken is...

		length ++ ;

		LineSegment chainSegment = linesegments[i];
		linesegments[i] = linesegments[ linesegments.size() - 1 ];
		linesegments.resize( linesegments.size() - 1 );

		if( length == 4 ) {
			chain.push_back( chainSegment );
			return;
		}

		if( !atStartPoint ) {
			chain.push_back( chainSegment );
		}
		// recursie!
		findChainOfLines( chainSegment, atStartPoint, linesegments, chain, length);
		if( atStartPoint ) {
			chain.push_back( chainSegment );
		}
		return;
	}
}

std::vector<LineSegment> EdgelDetector::findLinesWithCorners(std::vector<LineSegment> &linesegments) {
	std::vector<LineSegment> linesWithCorners;

	// detect corners. We expect black markers on a white background...
	for( int i=0, s = linesegments.size(); i<s; i++ ) {
		const int dx = static_cast<int>(linesegments[i].slope.x * 4.0f);
		const int dy = static_cast<int>(linesegments[i].slope.y * 4.0f);

		// check startpoint
		int x = linesegments[i].start.position.x - dx;
		int y = linesegments[i].start.position.y - dy;
		if( buffer->getPixel( x, y, 0 ) > WHITETRESHHOLD &&
		   buffer->getPixel( x, y, 1 ) > WHITETRESHHOLD &&
		   buffer->getPixel( x, y, 2 ) > WHITETRESHHOLD ) {
		   linesegments[i].start_corner = true;
		}

		// check endpoint
		x = linesegments[i].end.position.x + dx;
		y = linesegments[i].end.position.y + dy;
		if( buffer->getPixel( x, y, 0 ) > WHITETRESHHOLD &&
		   buffer->getPixel( x, y, 1 ) > WHITETRESHHOLD &&
		   buffer->getPixel( x, y, 2 ) > WHITETRESHHOLD ) {
		   linesegments[i].end_corner = true;
		}

		if( linesegments[i].start_corner || linesegments[i].end_corner ) {
			linesWithCorners.push_back( linesegments[i] );
		}
	}
	return linesWithCorners;
}


void EdgelDetector::extendLineSegments( std::vector<LineSegment> &lineSegments ) {
	// extend linesegments
	for( int i=0, s = lineSegments.size(); i<s; i++ ) {
		Vector2f startpoint, endpoint;
		extendLine( lineSegments[i].end.position, lineSegments[i].slope, lineSegments[i].end.slope, lineSegments[i].end.position, 999 );
		extendLine( lineSegments[i].start.position, -lineSegments[i].slope, lineSegments[i].end.slope, lineSegments[i].start.position, 999 );

	}
}


std::vector<Edgel> EdgelDetector::findEdgelsInRegion(const int left, const int top, const int width, const int height ) {
	float prev1, prev2;
	std::vector<Edgel> edgels;

// debug function
	if( drawSectorGrids ) {
		for( int y=top; y<top+height; y+=RASTERSIZE ) {
			drawLine( left, y, left+width, y, 128, 128, 128, 1);
		}
		for( int x=left; x<left+width; x+=RASTERSIZE ) {
			drawLine( x, top, x, top+height, 128, 128, 128, 1);
		}
	}

	// debug function
	if( drawSectors ) {
		drawLine( left, top, left+width, top, 255, 255, 0, 1);
		drawLine( left, top, left, top+height, 255, 255, 0, 1);
		drawLine( left, top+height, left+width, top+height, 255, 255, 0, 1);
		drawLine( left+width, top, left+width, top+height, 255, 255, 0, 1);
	}

	for( int y=0; y<height; y+=RASTERSIZE ) {
		unsigned char* offset = buffer->getBuffer() + (left + (y+top)*buffer->getWidth() ) * 3;
		prev1 = prev2 = 0.0f;
		const int pitch = 3;

		for( int x=0; x<width; x++) {
//// CODE DUPLICATED
			// check channels
			int current = edgeKernel( offset, pitch );
			if( current > THRESHOLD &&
				edgeKernel( offset+1, pitch ) > THRESHOLD &&
				edgeKernel( offset+2, pitch ) > THRESHOLD ) {
				// edge!
			} else {
				current = 0.0f;
			}
			// find local maximum
			if( prev1 > 0.0f && prev1 > prev2 && prev1 > current ) {
				Edgel edgel;
				edgel.setPosition(left+x-1, top+y);
				edgel.slope = edgeGradientIntensity(edgel.position.x, edgel.position.y);

				edgels.push_back(edgel);
			}
			prev2 = prev1;
			prev1 = current;

			offset += pitch;
//// CODE DUPLICATED
		}
	}


	// debug function
	if( drawEdges ) {
		for( int i=0, s = edgels.size(); i<s; i++ ) {
			drawPoint( edgels[i].position.x, edgels[i].position.y, 0, 0, 255, THICKNESS);
		}
	}
	int debugNumOfHorizontalEdges = edgels.size();

	for( int x=0; x<width; x+=RASTERSIZE ) {
		unsigned char* offset = buffer->getBuffer() + (left + x + (top*buffer->getWidth() )) * 3;
		const int pitch = 3*buffer->getWidth();
		prev1 = prev2 = 0.0f;

		for( int y=0; y<height; y++) {
//// CODE DUPLICATED
			// check channels
			int current = edgeKernel( offset, pitch );
			if( current > THRESHOLD &&
				edgeKernel( offset+1, pitch ) > THRESHOLD &&
				edgeKernel( offset+2, pitch ) > THRESHOLD ) {
				// edge!
			} else {
				current = 0.0f;
			}
			// find local maximum
			if( prev1 > 0.0f && prev1 > prev2 && prev1 > current  ) {
				Edgel edgel;

				edgel.setPosition(left+x, top+y-1);
				edgel.slope = edgeGradientIntensity(edgel.position.x, edgel.position.y);

				edgels.push_back(edgel);
			}
			prev2 = prev1;
			prev1 = current;

			offset += pitch;
//// CODE DUPLICATED
		}
	}

	// debug function
	if( drawEdges ) {
		for( int i=debugNumOfHorizontalEdges, s = edgels.size(); i<s; i++ ) {
			drawPoint( edgels[i].position.x, edgels[i].position.y, 0, 255, 0, THICKNESS);
		}
	}

	return edgels;
}


std::vector<LineSegment> EdgelDetector::findLineSegment(std::vector<Edgel> edgels) {
	std::vector<LineSegment> lineSegments;
	LineSegment lineSegmentInRun;

	srand(time(NULL));


	do {
		lineSegmentInRun.supportEdgels.clear();

		for (int i = 0; i < 25; i++) {
			Edgel r1;
			Edgel r2;

			const int max_iterations = 100;
			int iteration = 0, ir1, ir2;

			//pak 2 random edgels welke compatible zijn met elkaar.
			do {
				ir1 = (rand()%(edgels.size()));
				ir2 = (rand()%(edgels.size()));

				r1 = edgels.at(ir1);
				r2 = edgels.at(ir2);
				iteration++;
			} while ( ( ir1 == ir2 || !r1.isOrientationCompatible( r2 ) ) && iteration < max_iterations );

			if( iteration < max_iterations ) {
			// 2 edgels gevonden!
				LineSegment lineSegment;
				lineSegment.start = r1;
				lineSegment.end = r2;
				lineSegment.slope = r1.slope;

				//check welke edgels op dezelfde line liggen en voeg deze toe als support
				for (unsigned int o = 0; o < edgels.size(); o++) {
					if ( lineSegment.atLine( edgels.at(o) ) ) {
						lineSegment.addSupport( edgels.at(o) );
					}
				}

				if( lineSegment.supportEdgels.size() > lineSegmentInRun.supportEdgels.size() ) {
					lineSegmentInRun = lineSegment;
				}
			}
		}

		// slope van de line bepalen
		if( lineSegmentInRun.supportEdgels.size() >= EDGELSONLINE ) {
			float u1 = 0;
			float u2 = 50000;
			const Vector2f slope = (lineSegmentInRun.start.position - lineSegmentInRun.end.position);
			const Vector2f orientation = Vector2f( -lineSegmentInRun.start.slope.y, lineSegmentInRun.start.slope.x );

			if (abs (slope.x) <= abs(slope.y)) {
				for (std::vector<Edgel>::iterator it = lineSegmentInRun.supportEdgels.begin(); it!=lineSegmentInRun.supportEdgels.end(); ++it) {

					if ((*it).position.y > u1) {
						u1 = (*it).position.y;
						lineSegmentInRun.start = (*it);
					}

					if ((*it).position.y < u2) {
						u2 = (*it).position.y;
						lineSegmentInRun.end = (*it);
					}
				}
			} else {
				for (std::vector<Edgel>::iterator it = lineSegmentInRun.supportEdgels.begin(); it!=lineSegmentInRun.supportEdgels.end(); ++it) {

					if ((*it).position.x > u1) {
						u1 = (*it).position.x;
						lineSegmentInRun.start = (*it);
					}

					if ((*it).position.x < u2) {
						u2 = (*it).position.x;
						lineSegmentInRun.end = (*it);
					}
				}
			}

			// switch startpoint and endpoint according to orientation of edge

			if( dot( lineSegmentInRun.end.position - lineSegmentInRun.start.position, orientation ) < 0.0f ) {
				std::swap( lineSegmentInRun.start, lineSegmentInRun.end );
			}


			lineSegmentInRun.slope = (lineSegmentInRun.end.position - lineSegmentInRun.start.position).get_normalized();

			// heeft de lineSegmentInRun voldoende dan toevoegen aan lineSegments,
			// gebruikte edgels verwijderen..

			lineSegments.push_back( lineSegmentInRun );

			//TODO: Dit moet sneller!
			for(unsigned int i=0; i<lineSegmentInRun.supportEdgels.size(); i++) {
				for (std::vector<Edgel>::iterator it = edgels.begin(); it!=edgels.end(); ++it) {
					if( (*it).position.x == lineSegmentInRun.supportEdgels.at(i).position.x &&
						(*it).position.y == lineSegmentInRun.supportEdgels.at(i).position.y ) {
						edgels.erase( it );
						break;
					}
				}
			}
		}

	} while( lineSegmentInRun.supportEdgels.size() >= EDGELSONLINE && edgels.size() >= EDGELSONLINE );

	return lineSegments;
}

class LineSegmentDistance {
	public:
	LineSegmentDistance( float d, int i ) {
		distance = d;
		index = i;
	}
		float distance;
		int index;
};

inline bool operator<(const LineSegmentDistance& a, const LineSegmentDistance& b) {
	return a.distance < b.distance;
}

std::vector<LineSegment> EdgelDetector::mergeLineSegments(std::vector<LineSegment> linesegments, int max_iterations) {
	static std::vector<LineSegmentDistance> distanceIndex;

	for (int i = 0; i < linesegments.size(); i++) {
		LineSegment start = linesegments[i];

		distanceIndex.clear();

		// zoek alle lijnstukken waar je mee zou kunnen mergen...
		for (int j = 0; j < linesegments.size(); j++) {

			if(i != j
			    &&
			   // ze moeten dezelfde orientatie hebben ..
			    dot( linesegments[j].slope, start.slope ) > 0.99f
			    &&
			   // ze moeten in elkaars verlengde liggen, en dezelfde kant op 'wijzen'
			    dot( (linesegments[j].end.position - start.start.position).get_normalized(), start.slope ) > 0.99f
			   ) {

				// bereken distance tussen twee lijnen en sla op in index;
				const int squared_length = (linesegments[j].start.position - start.end.position).get_squared_length();

				if( squared_length < 25*25) {
					distanceIndex.push_back( LineSegmentDistance( squared_length , j ) );
				}
			}
		}
		if( !distanceIndex.size() ) {
			continue;
		}

		// sorteer op afstand

		std::sort( distanceIndex.begin(), distanceIndex.end() );

		// loop door alle lijnsegmenten waarmee je merged...

		for( std::vector<LineSegmentDistance>::iterator k = distanceIndex.begin(); k != distanceIndex.end(); k++ ) {

			const int j = (*k).index;

			const Vector2f startpoint = start.end.position;
			Vector2f endpoint = linesegments[j].start.position;


			const int length = (endpoint-startpoint).get_length();
			const Vector2f slope = (endpoint-startpoint).get_normalized();

			LineSegment between;
			between.start.position = startpoint;
			between.end.position = endpoint;


			if( extendLine( startpoint, slope, start.end.slope, endpoint, length ) ) {
				// i en j zijn gemerged...

				linesegments[i].end = linesegments[j].end;
				linesegments[i].slope = ( linesegments[i].end.position - linesegments[i].start.position ).get_normalized();

				linesegments[j].remove = true;
			} else {
				//between.drawLine( 255, 0, 0 );
				break;
			}
		}

		bool merged = false;

		// verwijder alle gemergede lijnstukken...
		for( int j=0; j < linesegments.size(); j++ ) {
			if( linesegments[j].remove ) {
				linesegments[j] = linesegments[ linesegments.size() - 1 ];
				linesegments.resize( linesegments.size() - 1 );
				j--;
				merged = true;
			}
		}
		if(merged) i = -1;
	}

	return linesegments;
}

bool EdgelDetector::extendLine( Vector2f startpoint, const Vector2f slope, const Vector2f gradient, Vector2f& endpoint, const int maxlength ) {
	const Vector2f normal = Vector2f( slope.y, -slope.x );
	bool merge = true;

	// controleer of de verbindingslijn wel op een rand ligt..
	for( int c=0; c<maxlength; c++ ) {
		startpoint += slope;

		if( EdgelDetector::edgeKernelX( startpoint.x, startpoint. y ) < THRESHOLD/2 &&
			EdgelDetector::edgeKernelY(  startpoint.x, startpoint. y ) < THRESHOLD/2 ) {
			merge = false;
			break;
		}

		if( EdgelDetector::edgeGradientIntensity(startpoint.x, startpoint.y) * gradient > 0.38f ) {
			continue;
		}
		if( EdgelDetector::edgeGradientIntensity(startpoint.x+normal.x, startpoint.y+normal.y) * gradient > 0.38f ) {
		//	startpoint += normal;
			continue;
		}
		if( EdgelDetector::edgeGradientIntensity(startpoint.x-normal.x, startpoint.y-normal.y) * gradient > 0.38f ) {
		//	startpoint -= normal;
			continue;
		}

		merge = false;
		break;

	}

	endpoint = startpoint - slope;
	return merge;
}

void EdgelDetector::setBuffer(Buffer* buffer) {
	this->buffer = buffer;
}