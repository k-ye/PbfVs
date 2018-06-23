#include "../include/obj_models_config_loader.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "../include/typedefs.h"
#include "../include/utils.h"

namespace pbf {

namespace {

void PlaceObjModel(point_t o, float scale, ObjModel *obj_model) {
  // Lower left
  point_t ll;
  ll.x = std::numeric_limits<float>::max();
  ll.y = std::numeric_limits<float>::max();
  ll.z = std::numeric_limits<float>::max();

  for (const point_t &v : obj_model->vertices) {
    ll.x = std::min(ll.x, v.x);
    ll.y = std::min(ll.y, v.y);
    ll.z = std::min(ll.z, v.z);
  }
  // std::cout << "lower-left: " <<  ll.x << ", " << ll.y << ", " << ll.z << std::endl;
  // std::cout << "new origin: " <<  o.x << ", " << o.y << ", " << o.z << std::endl;
  // std::cout << "scale: " <<  scale << std::endl;

  auto Compute = [=](float min_p, float vert_p, float o_p) -> float {
    return ((vert_p - min_p) * scale + o_p);
  };
  for (point_t &v : obj_model->vertices) {
    v.x = Compute(ll.x, v.x, o.x);
    v.y = Compute(ll.y, v.y, o.y);
    v.z = Compute(ll.z, v.z, o.z);
  }
}

} // namespace
std::vector<ObjModel> LoadModelsFromConfigFile(const std::string &filepath) {
  std::vector<ObjModel> result;
  auto f = [&](const std::string &line) {
    if (line.empty() || line[0] == '#') {
      return;
    }
    std::stringstream ss;
    ss << line;
    std::string obj_filepath;
    point_t o;
    float scale;
    ss >> obj_filepath >> o.x >> o.y >> o.z >> scale;

    ObjModel obj_model = LoadObjModel(obj_filepath);
    std::cout << obj_filepath << ", #vertices=" << obj_model.vertices.size()
        << ", #faces=" << obj_model.faces.size() << std::endl;
    PlaceObjModel(o, scale, &obj_model);
    result.push_back(std::move(obj_model));
  };
  ReadFileByLine(filepath, f);
  return result;
}
} // namespace pbf
