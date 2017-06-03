//
//  config.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/config.h"

#include "../include/utils.h"

namespace pbf {
	namespace detail_ {
		int Cast(const std::string& val_str, TypeTrait<int>) {
			return std::stoi(val_str);
		}

		long int Cast(const std::string& val_str, TypeTrait<long int>) {
			return std::stol(val_str);
		}

		unsigned Cast(const std::string& val_str, TypeTrait<unsigned>) {
			return (unsigned)std::stoul(val_str);
		}

		long unsigned Cast(const std::string& val_str, TypeTrait<long unsigned>) {
			return std::stoul(val_str);
		}

		float Cast(const std::string& val_str, TypeTrait<float>) {
			return std::stof(val_str);
		}
	} // namespace detail_

	void Config::Load(const char* filepath) {
		kv_map_.clear();

		std::stringstream ss;
		ss << ReadFile(filepath);
		std::string line;

		while (std::getline(ss, line)) {
			// a line that begins with '#' is a comment
			if (line.empty() || line[0] == '#')
				continue;
			size_t eq_pos = line.find('=');
			std::string key = line.substr(0, eq_pos);
			std::string val = line.substr(eq_pos + 1); // skip the '='

			kv_map_[key] = val;
		}

		filepath_ = filepath;
	}

	bool Config::GetOptionalImpl_(const std::string& key, std::string* val) const {
		auto it = kv_map_.find(key);
		if (it == kv_map_.end())
			return false;

		*val = it->second;
		return true;
	}
} // namespace pbf
