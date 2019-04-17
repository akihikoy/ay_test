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

#ifndef _h_buffer_
#define _h_buffer_ 1

class Buffer
{
public:
	inline int getWidth() { return m_nWidth; }
	inline int getHeight() { return m_nHeight; }
	inline unsigned char* getBuffer() { return m_pDataPtr; }

	void setBuffer(unsigned char* dataPtr, int width, int height);

	inline unsigned char getPixel(int x, int y, int channel) {
		x = x>0?x:0;
		y = y>0?y:0;

		x = x<getWidth()?x:getWidth()-1;
		y = y<getHeight()?y:getHeight()-1;

		int offset = ((x + (y*getWidth())) * 3)+channel;
		return getBuffer()[offset];
	}
	inline float getPixelColor(int x, int y, int channel) {
		return float( getPixel( x, y, channel) );
	}
	
private:
	int m_nWidth;
	int m_nHeight;
	unsigned char* m_pDataPtr;
};
#endif
