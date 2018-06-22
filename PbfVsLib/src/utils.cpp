//
//  utils.cpp
//  PBF
//
//  Created by Ye Kuang on 3/27/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/utils.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace pbf {
std::string ReadFile(const char *filepath) {
  std::ifstream fs(filepath, std::ios::in);
  std::stringstream ss;

  if (!fs.is_open()) {
    std::cerr << "Could not read file " << filepath << ". File does not exist."
              << std::endl;
    return "";
  }

  std::string line;
  while (!fs.eof()) {
    std::getline(fs, line);
    if (!line.empty()) {
      ss << line << '\n';
    }
  }

  fs.close();
  return ss.str();
}

} // namespace pbf
