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

std::function<int(float)> MakePosToGrid(float interval) {
  return [=](float pos) -> int {
    return (int)((pos - (interval * 0.5f)) / interval);
  };
}

std::function<float(int)> MakeGridToPos(float interval) {
  return [=](int grid) -> float {
    return (float)(grid * interval + (interval * 0.5f));
  };
}

std::function<bool(const glm::tvec3<int> &xyz)>
MakeIsInGrid(const glm::tvec3<int> &grid_sz) {
  return [=](const glm::tvec3<int> &xyz) -> bool {
    return ((0 <= xyz.x) && (xyz.x < grid_sz.x) && (0 <= xyz.y) &&
            (xyz.y < grid_sz.y) && (0 <= xyz.z) && (xyz.z < grid_sz.z));
  };
}

enum class CheckDirection { X, Y, Z };

class CheckDirHelper {
public:
  CheckDirHelper(CheckDirection flag) : flag_(flag) {}

  vec_t GetCheckDirection() const {
    switch (flag_) {
    case CheckDirection::X:
      return {1.0f, 0.0f, 0.0f};
    case CheckDirection::Y:
      return {0.0f, 1.0f, 0.0f};
    case CheckDirection::Z:
      return {0.0f, 0.0f, 1.0f};
    }
    // Should check fail.
    return vec_t(0.0f);
  }

  vec2 SliceCoordinate(const point_t &p) const {
    switch (flag_) {
    case CheckDirection::X:
      return {p.y, p.z};
    case CheckDirection::Y:
      return {p.x, p.z};
    case CheckDirection::Z:
      return {p.x, p.y};
    }
    // Should check fail.
    return vec2(0.0f);
  }

  float GetCheckedComponent(const point_t &p) const {
    switch (flag_) {
    case CheckDirection::X:
      return p.x;
    case CheckDirection::Y:
      return p.y;
    case CheckDirection::Z:
      return p.z;
    }
    // Should check fail.
    return 0.0f;
  }

  glm::tvec3<int> MapToAbsGridXyz(int c1, int c2, int c3) const {
    switch (flag_) {
    case CheckDirection::X:
      return {c3, c1, c2};
    case CheckDirection::Y:
      return {c1, c3, c2};
    case CheckDirection::Z:
      return {c1, c2, c3};
    }
    // Should check fail.
    return {0, 0, 0};
  };

private:
  const CheckDirection flag_;
};

void FillBitmaskByDir(const point_t &v1, const point_t &v2, const point_t &v3,
                      const glm::tvec3<int> &grid_sz, float interval,
                      const CheckDirHelper &check_dir_helper,
                      GridBitmask *flat_grid_bm) {
  auto PosToGrid = MakePosToGrid(interval);
  auto GridToPos = MakeGridToPos(interval);
  auto IsInGrid = MakeIsInGrid(grid_sz);

  const vec2 a = check_dir_helper.SliceCoordinate(v1);
  const vec2 b = check_dir_helper.SliceCoordinate(v2);
  const vec2 c = check_dir_helper.SliceCoordinate(v3);
  // Numerical precision issue, out of boundary issue. Ignore them...
  // Find the AABB of the projected triangle.
  const float min_c1 = std::min(a.x, std::min(b.x, c.x));
  const float max_c1 = std::max(a.x, std::max(b.x, c.x));
  const float min_c2 = std::min(a.y, std::min(b.y, c.y));
  const float max_c2 = std::max(a.y, std::max(b.y, c.y));

  int grid_c1_begin = PosToGrid(min_c1);
  if (GridToPos(grid_c1_begin) < min_c1 - kFloatEpsilon) {
    grid_c1_begin += 1;
  }
  int grid_c2_begin = PosToGrid(min_c2);
  if (GridToPos(grid_c2_begin) < min_c2 - kFloatEpsilon) {
    grid_c2_begin += 1;
  }

  glm::tvec2<int> grid_c12 = {grid_c1_begin, grid_c2_begin};
  vec2 cur_c12(0.0f);
  cur_c12.x = GridToPos(grid_c12.x);

  const vec_t v12 = v2 - v1;
  const vec_t v13 = v3 - v1;

  while (cur_c12.x < max_c1) {
    grid_c12.y = grid_c2_begin;
    cur_c12.y = GridToPos(grid_c12.y);
    while (cur_c12.y < max_c2) {
      vec2 uv(0.0f);
      if (IsInTriangle(a, b, c, cur_c12, &uv)) {
        point_t pt = v1 + v12 * uv.x + v13 * uv.y;
        // int grid_z = PosToGrid(pt.z);
        const int grid_c3 = PosToGrid(check_dir_helper.GetCheckedComponent(pt));
        const auto grid_xyz =
            check_dir_helper.MapToAbsGridXyz(grid_c12.x, grid_c12.y, grid_c3);
        if (IsInGrid(grid_xyz)) {
          size_t idx =
              ToFlatGridIndex(grid_xyz.x, grid_xyz.y, grid_xyz.z, grid_sz);
          (*flat_grid_bm)[idx] += 1;
        }
      }

      grid_c12.y += 1;
      cur_c12.y += interval;
    }
    cur_c12.x += 1;
    cur_c12.x += interval;
  }
}

void FillBitmask(const ObjModel &obj_model, const glm::tvec3<int> &grid_sz,
                 float interval, CheckDirection check_dir_flag,
                 GridBitmask *flat_grid_bm) {
  CheckDirHelper check_dir_helper(check_dir_flag);
  const auto &vertices = obj_model.vertices;
  const vec_t check_dir = check_dir_helper.GetCheckDirection();

  for (const auto &f : obj_model.faces) {
    point_t v1 = vertices[f.x];
    point_t v2 = vertices[f.y];
    point_t v3 = vertices[f.z];

    vec_t v12 = v2 - v1;
    vec_t v13 = v3 - v1;
    vec_t plane_norm = glm::cross(v12, v13);
    if (glm::abs(glm::dot(check_dir, plane_norm)) <= kFloatEpsilon) {
      // |check_dir| is in the surface, pass this face.
      continue;
    }
    FillBitmaskByDir(v1, v2, v3, grid_sz, interval, check_dir_helper,
                     flat_grid_bm);
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

  GridBitmask flat_grid_bm_x(grid_sz.x * grid_sz.y * grid_sz.z, 0);
  GridBitmask flat_grid_bm_y(grid_sz.x * grid_sz.y * grid_sz.z, 0);
  GridBitmask flat_grid_bm_z(grid_sz.x * grid_sz.y * grid_sz.z, 0);
  for (const ObjModel &obj_model : obj_models) {
    FillBitmask(obj_model, grid_sz, interval, CheckDirection::X,
                &flat_grid_bm_x);
    FillBitmask(obj_model, grid_sz, interval, CheckDirection::Y,
                &flat_grid_bm_y);
    FillBitmask(obj_model, grid_sz, interval, CheckDirection::Z,
                &flat_grid_bm_z);
  }

  for (int x = 0; x < grid_sz.x; ++x) {
    for (int y = 0; y < grid_sz.y; ++y) {
      for (int z = 0; z < grid_sz.z; ++z) {
        if (x > 0) {
          size_t front_idx = ToFlatGridIndex(x - 1, y, z, grid_sz);
          size_t cur_idx = ToFlatGridIndex(x, y, z, grid_sz);
          flat_grid_bm_x[cur_idx] += flat_grid_bm_x[front_idx];
        }
        if (y > 0) {
          size_t front_idx = ToFlatGridIndex(x, y - 1, z, grid_sz);
          size_t cur_idx = ToFlatGridIndex(x, y, z, grid_sz);
          flat_grid_bm_y[cur_idx] += flat_grid_bm_y[front_idx];
        }
        if (z > 0) {
          size_t front_idx = ToFlatGridIndex(x, y, z - 1, grid_sz);
          size_t cur_idx = ToFlatGridIndex(x, y, z, grid_sz);
          flat_grid_bm_z[cur_idx] += flat_grid_bm_z[front_idx];
        }
      }
    }
  }

  auto GridToPos = MakeGridToPos(interval);
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
        if ((flat_grid_bm_x[idx] & 1) && (flat_grid_bm_y[idx] & 1) &&
            (flat_grid_bm_z[idx] & 1)) {
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
