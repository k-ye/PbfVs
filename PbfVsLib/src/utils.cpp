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

std::string TrimLeft(const std::string &s, const std::string &matcher_str) {
  //  https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
  auto start_pos = s.find_first_not_of(matcher_str);
  if (start_pos != std::string::npos) {
    return s.substr(start_pos);
  }
  return "";
}

int ReadFileByLine(const std::string &filepath,
                   const std::function<void(size_t, const std::string &)> &f) {
  std::ifstream fs(filepath, std::ios::in);

  if (!fs.is_open()) {
    std::cerr << "Could not read file " << filepath << ". File does not exist."
              << std::endl;
    return -1;
  }

  std::string line;
  size_t line_no = 0;
  while (!fs.eof()) {
    std::getline(fs, line);
    f(line_no, line);
    ++line_no;
  }

  fs.close();
  return 0;
}

int ReadFileByLine(const std::string &filepath,
                   const std::function<void(const std::string &)> &f) {
  return ReadFileByLine(filepath,
                        [&f](size_t, const std::string &line) { f(line); });
}

std::string ReadFile(const std::string &filepath) {
  std::stringstream ss;
  auto f = [&ss](const std::string &line) {
    if (!line.empty()) {
      ss << line << '\n';
    }
  };
  ReadFileByLine(filepath, f);
  return ss.str();
}

} // namespace pbf
