//
//  point_drawer.cpp
//

#include "../include/point_drawer.h"

namespace pbf {

void AddPointToDraw(const glm::vec3 &pt, std::vector<GLfloat> *vertices,
                    std::vector<GLuint> * /*indices*/, const glm::vec3 &color,
                    float size) {
  vertices->push_back(pt[0]);
  vertices->push_back(pt[1]);
  vertices->push_back(pt[2]);

  vertices->push_back(color[0]);
  vertices->push_back(color[1]);
  vertices->push_back(color[2]);
}

void ChangePointToDraw(const glm::vec3 &pt, size_t index,
                       std::vector<GLfloat> *vertices, const glm::vec3 &color,
                       float /*size*/) {
  unsigned vidx = index * 6;
  (*vertices)[vidx] = pt[0];
  (*vertices)[vidx + 1] = pt[1];
  (*vertices)[vidx + 2] = pt[2];
  (*vertices)[vidx + 3] = color[0];
  (*vertices)[vidx + 4] = color[1];
  (*vertices)[vidx + 5] = color[2];
}
} // namespace pbf
