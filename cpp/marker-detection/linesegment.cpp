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

#include "linesegment.h"
#include <math.h>
//#include "bresenham.h"

void LineSegment::addSupport( Edgel& cmp ) {
	supportEdgels.push_back( cmp );
}
bool LineSegment::atLine( Edgel& cmp ) {	
	if( !start.isOrientationCompatible( cmp ) ) return false;

	// distance to line: (AB x AC)/|AB| 
	// A = r1
	// B = r2
	// C = cmp

	// AB ( r2.x - r1.x, r2.y - r1.y )
	// AC ( cmp.x - r1.x, cmp.y - r1.y )

	float cross = (float(end.position.x)-float(start.position.x)) *( float(cmp.position.y)-float(start.position.y));
	cross -= (float(end.position.y)-float(start.position.y)) *( float(cmp.position.x)-float(start.position.x));

	const float d1 = float(start.position.x)-float(end.position.x);
	const float d2 = float(start.position.y)-float(end.position.y);

	float distance = cross / Vector2f(d1, d2).get_length();

	return fabs(distance) < 0.75f;
}

bool LineSegment::isOrientationCompatible( LineSegment& cmp ) {	
	return  slope * cmp.slope > 0.92f; //0.38f; //cosf( 67.5f / 2 pi )
}

Vector2f LineSegment::getIntersection( LineSegment& b ) {
	Vector2f intersection;
	
	float denom = ((b.end.position.y - b.start.position.y)*(end.position.x - start.position.x)) -
					((b.end.position.x - b.start.position.x)*(end.position.y - start.position.y));
	
	float nume_a = ((b.end.position.x - b.start.position.x)*(start.position.y - b.start.position.y)) -
					((b.end.position.y - b.start.position.y)*(start.position.x - b.start.position.x));
	
	float ua = nume_a / denom;
	
	intersection.x = start.position.x + ua * (end.position.x - start.position.x);
	intersection.y = start.position.y + ua * (end.position.y - start.position.y);
	
	return intersection;
}