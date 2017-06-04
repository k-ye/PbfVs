#ifndef kernel_cuh
#define kernel_cuh

#include "typedefs.h"
#include "cuda_basic.h"
#include "shared_math.h"

namespace pbf {
namespace {
	CUDA_CALLABLE constexpr float kPoly6Factor() { return (315.0f / 64.0f / PI_FLT); }
	
	CUDA_CALLABLE constexpr float kSpikyGradFactor() { return (-45.0f / PI_FLT); }
} // namespace anonymous

	CUDA_CALLABLE inline float Poly6Value(const float s, const float h) {
		if (s < 0.0f || s >= h) return 0.0f;

		float x = (h * h - s * s) / (h * h * h);
		float result = kPoly6Factor() * x * x * x;
		return result;
	}
	
	CUDA_CALLABLE inline float Poly6Value(const float3 r, const float h) {
		float r_len = length(r);
		return Poly6Value(r_len, h);
	}
	
	CUDA_CALLABLE inline float3 SpikyGradient(const float3 r, const float h) {
		float r_len = length(r);
		if (r_len <= 0.0f || r_len >= h) return make_float3(0.0f);

		float x = (h - r_len) / (h * h * h);
		float g_factor = kSpikyGradFactor() * x * x;
		float3 result = normalize(r) * g_factor;
		return result;
	}
} // namespace pbf

#endif // kernel_cuh
