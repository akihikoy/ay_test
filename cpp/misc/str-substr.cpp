/*! \file    str-substr.cpp
    \brief   Str substr test.
    \author  Akihiko Yamaguchi, info@akihikoy.net
    \version 0.1
    \date    Feb 10, 2025.

g++ -g -Wall -O2 str-substr.cpp && ./a.out
*/
//-------------------------------------------------------------------------------------------
#include <iostream>
#include <string>

int main(int argc, char**argv)
{
  std::string img_depth_topic("/camera/aligned_depth_to_color/image_raw");

  std::string rs_name= img_depth_topic.substr(1,img_depth_topic.find('/',1)-1);

  std::cout<<"rs_name = "<<rs_name<<std::endl;
  return 0;
}
//-------------------------------------------------------------------------------------------
