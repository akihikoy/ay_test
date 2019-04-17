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

#ifndef _h_linesegment_
#define _h_linesegment_

#include "edgel.h"
#include <vector>

class LineSegment {
	public:

	LineSegment() : remove(false), start_corner(false), end_corner(false) {}

	bool atLine( Edgel& cmp );
	void addSupport( Edgel& cmp );
	bool isOrientationCompatible( LineSegment& cmp );
	Vector2f getIntersection( LineSegment& b );

	Edgel start, end;
	Vector2f slope;
	bool remove, start_corner, end_corner;

	std::vector<Edgel> supportEdgels;

	bool operator==(const LineSegment & rhs) const {
		return (start.position.x == rhs.start.position.x &&
				start.position.y == rhs.start.position.y &&
				end.position.x == rhs.end.position.x &&
				end.position.y == rhs.end.position.y
			);
	}
};
#endif