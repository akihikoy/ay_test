/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_THINING_HPP__
#define __OPENCV_THINING_HPP__

#include <opencv2/core/core.hpp>


namespace cv
{
namespace ximgproc
{

enum ThinningTypes{
    THINNING_ZHANGSUEN    = 0, // Thinning technique of Zhang-Suen
    THINNING_GUOHALL      = 1  // Thinning technique of Guo-Hall
};

//! @addtogroup ximgproc
//! @{

/** @brief Applies a binary blob thinning operation, to achieve a skeletization of the input image.

The function transforms a binary blob image into a skeletized form using the technique of Zhang-Suen.

@param src Source 8-bit single-channel image, containing binary blobs, with blobs having 255 pixel values.
@param dst Destination image of the same size and the same type as src. The function can work in-place.
@param thinningType Value that defines which thinning algorithm should be used. See cv::ximgproc::ThinningTypes
 */
CV_EXPORTS_W void thinning( InputArray src, OutputArray dst, int thinningType = THINNING_ZHANGSUEN);

//! @}

}
}

#endif // __OPENCV_THINING_HPP__
