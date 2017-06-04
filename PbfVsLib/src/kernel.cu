#include "../include/kernel.h"
#include "../include/kernel.cuh"
#include "../include/helper_math.h"
#include "../include/shared_math.h"

namespace pbf {
namespace {
	CUDA_CALLABLE constexpr float kPoly6Factor() { return (315.0f / 64.0f / PI_FLT); }
	
	CUDA_CALLABLE constexpr float kSpikyGradFactor() { return (-45.0f / PI_FLT); }
	
	CUDA_CALLABLE float Poly6ValueImpl(const float s, const float h) {
		if (s < 0.0f || s >= h) return 0.0f;

		float x = (h * h - s * s) / (h * h * h);
		float result = kPoly6Factor() * x * x * x;
		return result;
	}
	
	CUDA_CALLABLE float Poly6ValueImpl(const float3 r, const float h) {
		float r_len = length(r);
		return Poly6Value(r_len, h);
	}

	CUDA_CALLABLE float3 SpikyGradientImpl(const float3 r, const float h) {
		float r_len = length(r);
		if (r_len <= 0.0f || r_len >= h) return make_float3(0.0f);

		float x = (h - r_len) / (h * h * h);
		float g_factor = kSpikyGradFactor() * x * x;
		float3 result = normalize(r) * g_factor;
		return result;
	}
} // namespace anonymous

	float Poly6Value(const float s, const float h) { return Poly6ValueImpl(s, h); }
	float Poly6ValueGpu(const float s, const float h) { return Poly6ValueImpl(s, h); }
	
	float Poly6Value(const point_t& r, const float h) { return Poly6ValueImpl(Convert(r), h); }
	float Poly6ValueGpu(const float3 r, const float h) { return Poly6ValueImpl(r, h); }
	
	vec_t SpikyGradient(const point_t& r, const float h) { 
		return Convert(SpikyGradientImpl(Convert(r), h)); 
	}
	float3 SpikyGradientGpu(const float3 r, const float h) { return SpikyGradientImpl(r, h); }
} // namespace pbf