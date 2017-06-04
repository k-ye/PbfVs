#ifndef aabb_h
#define aabb_h

#include <algorithm>

#include "typedefs.h"

namespace pbf {
    
	class AABB {
    public:
        enum { X = 0, Y = 1, Z = 2, NDIM = 3 };
        
        AABB();
        AABB(const point_t& mn, const point_t& mx);
		AABB(const point_t& center, float half_size);
        
        point_t min() const { return min_; }
        point_t max() const { return max_; }
        point_t center() const { return (min_ + max_) * 0.5f; }
        
        float length(unsigned dim) const { return max_[dim] - min_[dim]; }
        float length_x() const { return max_.x - min_.x; }
        float length_y() const { return max_.y - min_.y; }
        float length_z() const { return max_.z - min_.z; }
        
        AABB& Inflate(const point_t& p);
        
        AABB& UnionWith(const AABB& other);
        
        bool OverlapsWith(const AABB& other) const;
        
        bool Contains(const AABB& other) const;
        
        bool Contains(const point_t& pt) const;
        
        static AABB UnionOf(const AABB& lhs, const AABB& rhs);
        
        static AABB IntersectionOf(const AABB& lhs, const AABB& rhs);
        
    private:
        point_t min_;
        point_t max_;
    }; // class AABB
} // namespace pbf
#endif /* aabb_h */
