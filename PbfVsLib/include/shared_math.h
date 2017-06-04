#ifndef shared_math_h
#define shared_math_h

#include "glm_headers.h"
#include "typedefs.h"

// value borrowed from glm::pi
#define PI_FLT (float)3.14159265358979323846264338327950288
#define kFloatEpsilon (float)1e-6

namespace pbf {
	template <typename T>
	T Interpolate(T val, T vmin, T vmax, T rmin, T rmax) {
		return (val - vmin) * (rmax - rmin) / (vmax - vmin) + rmin;
	}

} // namespace pbf

#endif // shared_math_h