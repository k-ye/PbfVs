//
//  arcball_camera.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/arcball_camera.h"

namespace pbf {
ArcballCamera::ArcballCamera()
    : arcball_radius_(4.0f), sensitivity_(2.0f),
      camera_pos_(0.0f, 0.0f, arcball_radius_), target_(0.0f, 0.0f, 0.0f) {}

void ArcballCamera::SetStageSize(float width, float height) {
  half_stage_size_[X] = width / 2.0f;
  half_stage_size_[Y] = height / 2.0f;

  if (width < height)
    stage_scale_ = 1.0f / width;
  else
    stage_scale_ = 1.0f / height;
}

void ArcballCamera::SetArcballRadius(float r) {
  arcball_radius_ = r;
  camera_pos_ = glm::vec3(0.0f, 0.0f, arcball_radius_);
}

void ArcballCamera::OnMouseLeftClick(float mx, float my) {
  SetLastMousePos_(mx, my);
  begin_arc_dir_ = ComputeArcballDir_(mx, my);
  cur_rot_ = last_rot_;
}

void ArcballCamera::OnMouseLeftDragging(float mx, float my) {
  auto cur_dir = ComputeArcballDir_(mx, my);

  float dir_cos = glm::dot(begin_arc_dir_, cur_dir);

  if (fabsf(1.0f - dir_cos) < FLT_EPSILON) {
    cur_rot_ = last_rot_;
  } else {
    float angle =
        glm::acos(std::max(std::min(dir_cos, 1.0f), -1.0f)) * sensitivity_;
    auto rot_axis = glm::cross(begin_arc_dir_, cur_dir);
    rot_axis = glm::normalize(rot_axis);

    glm::mat4 rot_mat;
    rot_mat = glm::rotate(rot_mat, angle, rot_axis);

    cur_rot_ = rot_mat * last_rot_;
  }
}

void ArcballCamera::OnMouseLeftRelease(float mx, float my) {
  last_rot_ = cur_rot_;
}

glm::mat4 ArcballCamera::GetViewMatrix() const {
  glm::mat4 trans;
  trans = glm::translate(trans, -target_);

  glm::vec3 zeros{0.0f, 0.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};

  glm::mat4 lookat = glm::lookAt(camera_pos_, zeros, up);

  glm::mat4 result = lookat * cur_rot_ * trans;
  return result;
}

void ArcballCamera::SetLastMousePos_(float mx, float my) {
  last_mouse_[X] = mx;
  last_mouse_[Y] = my;
}

glm::vec3 ArcballCamera::ComputeArcballDir_(float mousex, float mousey) {
  float bx = mousex - half_stage_size_[X];
  float by = mousey - half_stage_size_[Y];
  by = -by;

  bx *= stage_scale_;
  by *= stage_scale_;

  float bl = std::hypot(bx, by);

  if (bl > 1.0f) {
    bx /= bl;
    by /= bl;
    bl = 1.0f;
  }

  float bz = std::sqrt(1.0f - bl * bl);
  return {bx, by, bz};
}
} // namespace pbf
