#ifndef obj_models_config_loader_h
#define obj_models_config_loader_h

#include <string>
#include <vector>

#include "obj_model.h"

namespace pbf {

std::vector<ObjModel> LoadModelsFromConfigFile(const std::string &filepath);

} // namespace pbf

#endif // obj_models_config_loader_h
