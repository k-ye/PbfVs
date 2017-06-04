#include "../include/kernel.h"

namespace pbf {
	float Poly6Value(const point_t& r, const float h) { return Poly6Value(Convert(r), h); }
	
	vec_t SpikyGradient(const point_t& r, const float h) { 
		return Convert(SpikyGradient(Convert(r), h)); 
	}

} // namespace pbf