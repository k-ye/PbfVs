#ifndef arcball_camera_h
#define arcball_camera_h

#include "glm_headers.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace pbf {
class ArcballCamera {
private:
  enum { X = 0, Y = 1 };

public:
  ArcballCamera();

  ArcballCamera(const ArcballCamera &) = delete;
  ArcballCamera &operator=(const ArcballCamera &) = delete;

  void SetStageSize(float width, float height);

  void SetSensitivity(float s) { sensitivity_ = s; }

  float GetArcballRadius() const { return arcball_radius_; }

  void SetArcballRadius(float r);

  void OnMouseLeftClick(float mx, float my);

  void OnMouseLeftDragging(float mx, float my);

  void OnMouseLeftRelease(float mx, float my);

  glm::mat4 GetViewMatrix() const;

private:
  void SetLastMousePos_(float mx, float my);

  glm::vec3 ComputeArcballDir_(float mousex, float mousey);

private:
  // stage
  float half_stage_size_[2];
  float stage_scale_;

  float arcball_radius_;
  float sensitivity_;
  glm::vec3 camera_pos_;
  glm::vec3 target_;

  // glm::matN() by default creates the identity matrix of N x N
  glm::mat4 cur_rot_;
  glm::mat4 last_rot_;

  float last_mouse_[2];
  glm::vec3 begin_arc_dir_;
};
} // namespace pbf

#endif // arcball_cameca_h
