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

#include "edgel.h"

bool Edgel::isOrientationCompatible( Edgel& cmp ) {	
	//return fabs( cmp.orientation - orientation ) < 0.0675;

	return slope * cmp.slope > 0.38f; //cosf( 67.5f / 2 pi ) ; //
}

void Edgel::setPosition(int x, int y) {
	position = Vector2f(x,y);
}
