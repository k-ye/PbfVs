#ifndef kernel_h
#define kernel_h

#include "kernel.cuh"

namespace pbf {
	float Poly6Value(const point_t& r, const float h);
	
	vec_t SpikyGradient(const point_t& r, const float h);
} // namespace pbf

#endif // kernel_h 
