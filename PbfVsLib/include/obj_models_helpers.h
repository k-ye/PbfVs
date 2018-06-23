#ifndef obj_models_helpers_h
#define obj_models_helpers_h

#include <string>
#include <vector>

#include "obj_model.h"

namespace pbf {

std::vector<ObjModel> LoadModelsFromConfigFile(const std::string &filepath);

std::vector<point_t>
FillPointsInObjModels(const std::vector<ObjModel> &obj_models,
                      glm::vec3 world_sz, float interval);
} // namespace pbf

#endif // obj_models_config_loader_h
