#ifndef config_h
#define config_h

#include "typedefs.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pbf {
namespace detail_ {

////////////////////////////////////////////////////
// Casters from std::string to some type.

inline std::string Cast(const std::string &val_str, TypeTrait<std::string>) {
  return val_str;
}

int Cast(const std::string &val_str, TypeTrait<int>);

long int Cast(const std::string &val_str, TypeTrait<long int>);

unsigned Cast(const std::string &val_str, TypeTrait<unsigned>);

long unsigned Cast(const std::string &val_str, TypeTrait<long unsigned>);

float Cast(const std::string &val_str, TypeTrait<float>);

// entry point for all casters
template <typename T> T Cast(const std::string &val_str) {
  return Cast(val_str, TypeTrait<T>{});
}

////////////////////////////////////////////////////

} // namespace detail_

// This class reads a very simple config file format that contains only
// key-value pairs without any scope concept.
//
// Sample format:
//
// # Line comment begins with a '#'.
// # Key-value pairs are separated by '=', no whitespace allowed.
// # Empty lines can be inserted freely to better illustrate the structure
// # of the file.
// key1=val1
// key2=val2
//
// key3=val2
// # ...
class Config {
public:
  // Load a config file in |filepath|
  void Load(const char *filepath);

  // Get the value of |key| optionally. Returns false if |key| is not found.
  // Otherwise stores the value in |result| and returns true.
  template <typename T>
  bool GetOptional(const std::string &key, T *result) const {
    std::string val_str;
    bool found = GetOptionalImpl_(key, &val_str);

    if (!found)
      return false;
    *result = detail_::Cast<T>(val_str);
    return true;
  }

  // Get the value of |key|. Throws std::runtime_error if |key| is not found.
  template <typename T> T Get(const std::string &key) const {
    T result;
    bool found = GetOptional(key, &result);

    if (!found) {
      std::stringstream ss;
      ss << "Cannot find key: " << key << " in the config file: " << filepath_;
      throw std::runtime_error(ss.str());
    }

    return result;
  }

  // Add a new key-value pair to the config.
  template <typename T> void Set(const std::string &key, const T &val) {
    kv_map_[key] = std::to_string(val);
  }

private:
  bool GetOptionalImpl_(const std::string &key, std::string *val) const;

private:
  std::string filepath_;
  std::unordered_map<std::string, std::string> kv_map_;
};
} // namespace pbf

#endif /* config_h */
