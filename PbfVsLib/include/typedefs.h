#ifndef pbf_typedefs_h
#define pbf_typedefs_h

#include "glm_headers.h"
#include "helper_math.h"

namespace pbf {
    typedef glm::vec3 point_t;
    typedef glm::vec3 vec_t;
	typedef float3 d_point_t;
	typedef float3 d_vec_t;

	float3 Convert(const glm::vec3& v);
	glm::vec3 Convert(const float3& v);

    ////////////////////////////////////////////////////
	// Trait helpers
	template <typename T>
    struct TypeTrait { typedef T type; };

	template <bool cond, typename Then, typename Else>
	struct IfElse { typedef Else type; };

	template <typename Then, typename Else>
	struct IfElse<true, Then, Else> { typedef Then type; };
} // namespace pbf

#endif // pbf_typedefs_h