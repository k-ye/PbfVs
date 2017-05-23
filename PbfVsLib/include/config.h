//
//  config.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef config_h
#define config_h

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pbf
{
namespace detail_
{
    template <typename T>
    struct TypeTrait { typedef T type; };
    
    ////////////////////////////////////////////////////
	// Casters from std::string to some type.

    inline std::string Cast(const std::string& val_str, TypeTrait<std::string>) {
        return val_str;
    }
    
	int Cast(const std::string& val_str, TypeTrait<int>);
    
	long int Cast(const std::string& val_str, TypeTrait<long int>);
    
	unsigned Cast(const std::string& val_str, TypeTrait<unsigned>);
    
	long unsigned Cast(const std::string& val_str, TypeTrait<long unsigned>);
    
	float Cast(const std::string& val_str, TypeTrait<float>);
    
    template <typename T>
    T Cast(const std::string& val_str) { return Cast(val_str, TypeTrait<T>{}); }
    
    ////////////////////////////////////////////////////
    
    
} // namespace detail_
    
    class Config
    {
    public:
        void Load(const char* filepath);
        
        template <typename T>
        bool GetOptional(const std::string& key, T* result) const {
            std::string val_str;
            bool found = GetOptionalImpl_(key, &val_str);
            
            if (!found)
                return false;
            *result = detail_::Cast<T>(val_str);
            return true;
        }
        
        template <typename T>
        T Get(const std::string& key) const {
            T result;
            bool found = GetOptional(key, &result);
            
            if (!found) {
                std::stringstream ss;
                ss << "Cannot find key: " << key
                    << " in the config file: " << filepath_;
                throw std::runtime_error(ss.str());
            }
            
            return result;
        }
        
        template <typename T>
        void Set(const std::string& key, const T& val) {
            kv_map_[key] = std::to_string(val);
        }
        
    private:
        bool GetOptionalImpl_(const std::string& key, std::string* val) const;
        
    private:
        std::string filepath_;
        std::unordered_map<std::string, std::string> kv_map_;
        
    };
} // namespace pbf

#endif /* config_h */
