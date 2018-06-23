#include "../include/obj_models_helpers.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "../include/aabb.h"
#include "../include/shared_math.h"
#include "../include/typedefs.h"
#include "../include/utils.h"

namespace pbf {
namespace {

using GridBitmask = std::vector<int>;
using glm::vec2;

point_t FindLowerBound(const ObjModel &obj_model) {
  point_t lb;
  lb.x = std::numeric_limits<float>::max();
  lb.y = std::numeric_limits<float>::max();
  lb.z = std::numeric_limits<float>::max();
  for (const point_t &v : obj_model.vertices) {
    lb.x = std::min(lb.x, v.x);
    lb.y = std::min(lb.y, v.y);
    lb.z = std::min(lb.z, v.z);
  }
  return lb;
}

point_t FindUpperBound(const ObjModel &obj_model) {
  point_t ub;
  ub.x = std::numeric_limits<float>::lowest();
  ub.y = std::numeric_limits<float>::lowest();
  ub.z = std::numeric_limits<float>::lowest();
  for (const point_t &v : obj_model.vertices) {
    ub.x = std::max(ub.x, v.x);
    ub.y = std::max(ub.y, v.y);
    ub.z = std::max(ub.z, v.z);
  }
  return ub;
}
void PlaceObjModel(point_t o, float scale, ObjModel *obj_model) {
  // Lower left
  point_t ll = FindLowerBound(*obj_model);
  auto Compute = [=](float min_p, float vert_p, float o_p) -> float {
    return ((vert_p - min_p) * scale + o_p);
  };
  for (point_t &v : obj_model->vertices) {
    v.x = Compute(ll.x, v.x, o.x);
    v.y = Compute(ll.y, v.y, o.y);
    v.z = Compute(ll.z, v.z, o.z);
  }
}

bool IsInTriangle(vec2 a, vec2 b, vec2 c, vec2 p, vec2 *uv) {
  // Ctrl+C/V: http://blackpawn.com/texts/pointinpoly/default.html
  using glm::dot;

  vec2 v0 = c - a;
  vec2 v1 = b - a;
  vec2 v2 = p - a;

  float d00 = dot(v0, v0);
  float d01 = dot(v0, v1);
  float d02 = dot(v0, v2);
  float d11 = dot(v1, v1);
  float d12 = dot(v1, v2);

  float inv_denom = 1.0f / (d00 * d11 - d01 * d01);
  float u = (d11 * d02 - d01 * d12) * inv_denom;
  float v = (d00 * d12 - d01 * d02) * inv_denom;

  if ((0 <= u) && (0 <= v) && (u + v < 1.0f)) {
    // Point is in the triangle.
    if (uv) {
      uv->x = u;
      uv->y = v;
    }
    return true;
  }
  return false;
}

size_t ToFlatGridIndex(int x, int y, int z, const glm::tvec3<int> &grid_sz) {
  size_t idx = (x * grid_sz.y + y) * grid_sz.z + z;
  return idx;
}

void FillBitmask(const ObjModel &obj_model, const glm::tvec3<int> &grid_sz,
                 float interval, GridBitmask *flat_grid_bm) {
  const auto &vertices = obj_model.vertices;
  const vec_t z_dir = {0.0f, 0.0f, 1.0f};
  const float half_interval = interval * 0.5f;

  auto PosToGrid = [=](float pos) -> int {
    return (int)((pos - half_interval) / interval);
  };
  auto GridToPos = [=](int grid) -> float {
    return (float)(grid * interval + half_interval);
  };
  auto IsInGrid = [&](int x, int y, int z) -> bool {
    return ((0 <= x) && (0 <= y) && (0 <= z) && (x < grid_sz.x) &&
            (y < grid_sz.y) && (z < grid_sz.z));
  };

  for (const auto &f : obj_model.faces) {
    point_t v1 = vertices[f.x];
    point_t v2 = vertices[f.y];
    point_t v3 = vertices[f.z];

    vec_t v12 = v2 - v1;
    vec_t v13 = v3 - v1;
    vec_t plane_norm = glm::cross(v12, v13);
    if (glm::abs(glm::dot(z_dir, plane_norm)) <= kFloatEpsilon) {
      // the z axis is in the surface, pass this face.
      continue;
    }
    vec2 a = {v1.x, v1.y};
    vec2 b = {v2.x, v2.y};
    vec2 c = {v3.x, v3.y};
    // Numerical precision issue, out of boundary issue. Ignore them...
    // Find the AABB of the XY-projected triangle.
    float min_x = std::min(a.x, std::min(b.x, c.x));
    float min_y = std::min(a.y, std::min(b.y, c.y));
    float max_x = std::max(a.x, std::max(b.x, c.x));
    float max_y = std::max(a.y, std::max(b.y, c.y));

    int grid_x_begin = PosToGrid(min_x);
    if (GridToPos(grid_x_begin) < min_x - kFloatEpsilon) {
      grid_x_begin += 1;
    }
    int grid_y_begin = PosToGrid(min_y);
    if (GridToPos(grid_y_begin) < min_y - kFloatEpsilon) {
      grid_y_begin += 1;
    }

    glm::tvec2<int> grid_xy = {grid_x_begin, grid_y_begin};
    vec2 cur_xy(0.0f);
    cur_xy.x = GridToPos(grid_xy.x);
    while (cur_xy.x < max_x) {
      grid_xy.y = grid_y_begin;
      cur_xy.y = GridToPos(grid_xy.y);
      while (cur_xy.y < max_y) {
        vec2 uv(0.0f);
        if (IsInTriangle(a, b, c, cur_xy, &uv)) {
          point_t pt = v1 + v12 * uv.x + v13 * uv.y;
          int grid_z = PosToGrid(pt.z);
          if (IsInGrid(grid_xy.x, grid_xy.y, grid_z)) {
            size_t idx = ToFlatGridIndex(grid_xy.x, grid_xy.y, grid_z, grid_sz);
            (*flat_grid_bm)[idx] += 1;
          }
        }

        grid_xy.y += 1;
        cur_xy.y += interval;
      }
      cur_xy.x += 1;
      cur_xy.x += interval;
    }
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

std::vector<point_t>
FillPointsInObjModels(const std::vector<ObjModel> &obj_models,
                      glm::vec3 world_sz, float interval) {
  // Algorithm: Eisemann, Elmar, and Xavier Decoret. "Single-pass GPU solid
  // oxelization for real-time applications."
  glm::tvec3<int> grid_sz;
  grid_sz.x = (int)(world_sz.x / interval) - 1;
  grid_sz.y = (int)(world_sz.y / interval) - 1;
  grid_sz.z = (int)(world_sz.z / interval) - 1;

  GridBitmask flat_grid_bm(grid_sz.x * grid_sz.y * grid_sz.z, 0);
  for (const ObjModel &obj_model : obj_models) {
    FillBitmask(obj_model, grid_sz, interval, &flat_grid_bm);
  }

  for (int x = 0; x < grid_sz.x; ++x) {
    for (int y = 0; y < grid_sz.y; ++y) {
      for (int z = 1; z < grid_sz.z; ++z) {
        size_t front_idx = ToFlatGridIndex(x, y, z - 1, grid_sz);
        size_t cur_idx = ToFlatGridIndex(x, y, z, grid_sz);
        flat_grid_bm[cur_idx] += flat_grid_bm[front_idx];
      }
    }
  }

  // TODO(k-ye): duplicate function;
  float half_interval = interval * 0.5f;
  auto GridToPos = [=](int grid) -> float {
    return (float)(grid * interval + half_interval);
  };

  std::vector<AABB> obj_aabbs;
  for (const auto &obj_model : obj_models) {
    AABB aabb(FindLowerBound(obj_model), FindUpperBound(obj_model));
    obj_aabbs.push_back(aabb);
  }

  std::vector<point_t> result;
  for (int x = 0; x < grid_sz.x; ++x) {
    for (int y = 0; y < grid_sz.y; ++y) {
      for (int z = 0; z < grid_sz.z; ++z) {
        size_t idx = ToFlatGridIndex(x, y, z, grid_sz);
        if (flat_grid_bm[idx] & 1) {
          point_t pt;
          pt.x = GridToPos(x);
          pt.y = GridToPos(y);
          pt.z = GridToPos(z);
          bool in_aabb = false;
          for (const AABB &aabb : obj_aabbs) {
            if (aabb.Contains(pt)) {
              in_aabb = true;
              break;
            }
          }
          if (in_aabb) {
            // std::cout << x << ", " << y << ", " << z << '\n';
            result.push_back(pt);
          }
        }
      }
    }
  }
  return result;
}

} // namespace pbf
