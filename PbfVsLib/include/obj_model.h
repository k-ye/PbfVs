#ifndef obj_model_h
#define obj_model_h

#include <string>
#include <vector>

#include "glm_headers.h"
#include "typedefs.h"

namespace pbf {

struct ObjModel {
  // A face is a triangle that connects three vertices;
  using face_t = glm::tvec3<size_t>;

  std::vector<point_t> vertices;
  std::vector<face_t> faces;
};

ObjModel LoadObjModel(const std::string &filepath);

} // namespace pbf
#endif // obj_model_h
