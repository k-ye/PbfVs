#include "../include/typedefs.h"

namespace pbf {
float3 Convert(const glm::vec3 &v) { return make_float3(v.x, v.y, v.z); }
glm::vec3 Convert(const float3 &v) { return glm::vec3{v.x, v.y, v.z}; }
} // namespace pbf
