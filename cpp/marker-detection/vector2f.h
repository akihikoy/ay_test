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

#ifndef _h_vector2f_
#define _h_vector2f_

#include <math.h>

//==============================================================================
//  Vector2f
//==============================================================================
class Vector2f {
public:
    // constructors
    Vector2f(void) {}
    Vector2f(float new_x, float new_y) { set(new_x, new_y); }
    Vector2f(const float* rhs)         { x=*rhs; y=*(rhs+1); }
    Vector2f(const Vector2f& rhs)      { set(rhs.x, rhs.y); }
    Vector2f(float val)                { set_to(val); }

    ~Vector2f() {}

    inline void set_to(float val) { x = y = val; }
    inline void set(float new_x, float new_y) { x=new_x; y=new_y; }

    // vector algebra
    inline float get_length() const { return (float)sqrt(get_squared_length()); }
    inline float get_squared_length() const { return (x*x)+(y*y); }

    inline void normalize() {
        const float inv_length = 1.0f / get_length();
        (*this) *= inv_length;
    }
    Vector2f get_normalized() const {
        Vector2f result(*this);
        result.normalize();
        return result;
    }
    inline void fast_normalize() {
        const float inv_length = fast_inv_sqrt( get_squared_length() );
        (*this) *= inv_length;
    }
    Vector2f get_fast_normalized() const {
        Vector2f result(*this);
        result.fast_normalize();
        return result;
    }

	//This code is purported to be from Quake3 and provides an approximation
	//to the inverse square root of a number, 1/sqrtf(x).
	inline float fast_inv_sqrt(float x) {
		float xhalf = 0.5f*x;
		long i = *(long*) &x;
	//    i = 0x5f3759df - (i>>1);    // original
	//    i = 0x5f375a86 - (i>>1);  // better?
		i = 0x5F400000 - (i>>1);  // Nick (flipcode)?
		x = *(float*) &i;
		x = x*(1.5f-xhalf*x*x);        // Newton step, repeating increases accuracy
	//    x = x*(1.5f-xhalf*x*x);        // Newton step, repeating increases accuracy

		return x;
	}


    // overloaded operators
    Vector2f operator+(const Vector2f & rhs) const { return Vector2f(x + rhs.x, y + rhs.y); }
    Vector2f operator-(const Vector2f & rhs) const { return Vector2f(x - rhs.x, y - rhs.y); }
    Vector2f operator/(const float rhs)      const { return (rhs==0.0f) ? Vector2f(0.0f, 0.0f) : Vector2f(x / rhs, y / rhs); }
    Vector2f operator*(const float rhs)      const { return Vector2f(x*rhs, y*rhs); }
    float operator*(const Vector2f & rhs)    const { return x*rhs.x + y*rhs.y; }
    friend Vector2f operator*(float scale_factor, const Vector2f & rhs) { return rhs * scale_factor; }

    bool operator==(const Vector2f & rhs) const { return (x==rhs.x && y==rhs.y); }
    bool operator!=(const Vector2f & rhs) const { return !((*this)==rhs); }

    void operator+=(const Vector2f & rhs) { x+=rhs.x; y+=rhs.y; }
    void operator-=(const Vector2f & rhs) { x-=rhs.x; y-=rhs.y; }
    void operator*=(const float rhs)      { x*=rhs; y*=rhs; }
    void operator/=(const float rhs)      { if(rhs!=0.0f) { x/=rhs; y/=rhs; } }

    Vector2f operator-(void) const {return Vector2f(-x, -y);}
    Vector2f operator+(void) const {return *this;}

    operator float* () const {return (float*) this;}

    // member variables
    float x;
    float y;
};

static inline Vector2f lerp(const Vector2f & v1, const Vector2f & v2, float factor) {
    return v1*(1.0f-factor) + v2*factor;
}

static inline float dot(const Vector2f & lhs, const Vector2f & rhs) {
    return (lhs.x * rhs.x +
            lhs.y * rhs.y );
}

inline bool is_finite(const Vector2f & v1) {
    return (is_finite(v1.x) && is_finite(v1.y));
}

static inline float lerp(const float &v1, const float &v2, float factor) {
    return v1*(1.0f-factor) + v2*factor;
}

#endif