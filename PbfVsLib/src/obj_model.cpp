#include "../include/obj_model.h"

#include "../include/utils.h"
#include <sstream>

namespace pbf {

ObjModel LoadObjModel(const std::string &filepath) {
  ObjModel result;
  auto vert_f = [&result](const std::string &line) {
    std::string l = TrimLeft(line);
    if (l.empty() || l[0] != 'v') {
      return;
    }
    std::stringstream ss;
    ss << l;
    std::string junk; // consume 'v'
    ss >> junk;
    point_t v;
    ss >> v.x >> v.y >> v.z;
    // v.x *= 100.0f;
    // v.y *= 100.0f;
    // v.z *= 100.0f;
    result.vertices.push_back(v);
  };
  // Read the vertices in the second pass.
  ReadFileByLine(filepath, vert_f);

  bool ok = true;
  size_t num_vertices = result.vertices.size();
  auto face_f = [&ok, &result, num_vertices](const std::string &line) {
    if (!ok) {
      return;
    }
    std::string l = TrimLeft(line);
    if (l.empty() || l[0] != 'f') {
      return;
    }
    std::stringstream ss;
    ss << l;
    std::string junk; // consume 'f'
    ss >> junk;
    ObjModel::face_t f;
    ss >> f.x >> f.y >> f.z;
    f.x -= 1;
    f.y -= 1;
    f.z -= 1;
    if (f.x >= num_vertices || f.y >= num_vertices || f.z >= num_vertices) {
      ok = false;
      return;
    }
    result.faces.push_back(f);
  };
  // Read the faces in the first pass.
  ReadFileByLine(filepath, face_f);
  if (!ok) {
    // TODO: should have a better way to surface the error.
    return {};
  }
  return result;
}
} // namespace pbf
