#include <stdexcept>

#include "../include/aabb.h"

namespace pbf
{
    AABB::AABB() : min_(0.0f), max_(0.0f) { }
    
    AABB::AABB(const point_t& mn, const point_t& mx)
		: min_(mn) , max_(mx) { }
	
	AABB::AABB(const point_t& center, float half_size) {
		point_t half_vec{ half_size };
		min_ = center - half_vec;
		max_ = center + half_vec;
	}
    
    AABB& AABB::Inflate(const point_t& p) {
        for (int d = 0; d < NDIM; ++d) {
            min_[d] = p[d];
            max_[d] += p[d];
        }
        return *this;
    }
    
    AABB& AABB::UnionWith(const AABB& other) {
        for (int d = 0; d < NDIM; ++d) {
            min_[d] = std::min(min_[d], other.min_[d]);
            max_[d] = std::max(max_[d], other.max_[d]);
        }
        
        return *this;
    }
    
    bool AABB::OverlapsWith(const AABB& other) const {
        bool ncond = (other.max_[X] < min_[X]) || (max_[X] < other.min_[X]) ||
        (other.max_[Y] < min_[Y]) || (max_[Y] < other.min_[Y]) ||
        (other.max_[Z] < min_[Z]) || (max_[Z] < other.min_[Z]);
        return !ncond;
    }
    
    bool AABB::Contains(const AABB& other) const {
        bool cond = (min_[X] <= other.min_[X]) && (max_[X] <= max_[X]) &&
        (min_[Y] <= other.min_[Y]) && (other.max_[Y] <= max_[Y]) &&
        (min_[Z] <= other.min_[Z]) && (other.max_[Z] <= max_[Z]);
        return cond;
    }
    
    bool AABB::Contains(const point_t& pt) const {
        bool cond = (min_[X] <= pt[X]) && (pt[X] <= max_[X]) &&
        (min_[Y] <= pt[Y]) && (pt[Y] <= max_[Y]) &&
        (min_[Z] <= pt[Z]) && (pt[Z] <= max_[Z]);
        return cond;
    }
    
    AABB AABB::UnionOf(const AABB& lhs, const AABB& rhs) {
        AABB result = lhs;
        result.UnionWith(rhs);
        return result;
    }
    
    AABB AABB::IntersectionOf(const AABB& lhs, const AABB& rhs) {
        if (!lhs.OverlapsWith(rhs)) {
            throw std::runtime_error("The two AABB does not intersect");
        }
        
        AABB result = lhs;
        for (int d = 0; d < NDIM; ++d) {
            result.min_[d] = std::max(result.min_[d], rhs.min_[d]);
            result.max_[d] = std::min(result.max_[d], rhs.max_[d]);
        }
        return result;
    }
} // namespace pbf
