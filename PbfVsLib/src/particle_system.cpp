//
//  particle_system.cpp
//  PBF
//
//  Created by Ye Kuang on 3/31/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/particle_system.h"

namespace pbf {
////////////////////////////////////////////////////

ParticleSystem::Particle::Particle(ParticleSystem *ps, index_t i)
    : ps_(ps), index_(i) {}

const point_t &ParticleSystem::Particle::position() const {
  return ps_->particles_[index_].position;
}

void ParticleSystem::Particle::set_position(const point_t &p) {
  ps_->particles_[index_].position = p;
}

const point_t &ParticleSystem::Particle::velocity() const {
  return ps_->particles_[index_].velocity;
}

void ParticleSystem::Particle::set_velocity(const point_t &v) {
  ps_->particles_[index_].velocity = v;
}

////////////////////////////////////////////////////

ParticleSystem::Particle ParticleSystem::Get(index_t i) const {
  return {const_cast<ParticleSystem *>(this), i};
}

ParticleSystem::Particle ParticleSystem::Add(const point_t &p,
                                             const point_t &v) {
  index_t pi = particles_.size();
  particles_.push_back({p, v});
  return Get(pi);
}

ParticleSystem::index_t ParticleSystem::Add(size_t n, const point_t &p,
                                            const point_t &v) {
  index_t pi = particles_.size();
  particles_.insert(particles_.end(), n, {p, v});
  return pi;
}
} // namespace pbf
