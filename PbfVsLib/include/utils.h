//
//  utils.h
//  PBF
//
//  Created by Ye Kuang on 10/14/16.
//  Copyright © 2016 Ye Kuang. All rights reserved.
//

#ifndef utils_h
#define utils_h

#include <fstream>
#include <functional>
#include <iostream>
#include <string>

namespace pbf {

std::string TrimLeft(const std::string &s, const std::string &matcher_str=" \t");

int ReadFileByLine(const std::string &filepath,
                   const std::function<void(const std::string &)> &f);

std::string ReadFile(const std::string &filepath);

} // namespace pbf

#endif /* utils_h */
