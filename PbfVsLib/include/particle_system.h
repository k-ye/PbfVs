//
//  particle_system.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef particle_system_h
#define particle_system_h

#include "typedefs.h"

#include <vector>

namespace pbf {
class ParticleSystem {
public:
  typedef size_t index_t;

public:
  class Particle;
  friend class Particle;

  class Particle {
  public:
    Particle(const Particle &p) = default;
    Particle &operator=(const Particle &p) = default;

    const point_t &position() const;
    void set_position(const point_t &p);

    const point_t &velocity() const;
    void set_velocity(const point_t &v);

    index_t index() const { return index_; }

  private:
    Particle(ParticleSystem *ps, index_t i);

    friend class ParticleSystem;

  private:
    ParticleSystem *ps_;
    const index_t index_;
  }; // class Particle

  inline size_t NumParticles() const { return particles_.size(); }

  Particle Get(index_t i) const;

  Particle Add(const point_t &p, const point_t &v);

  index_t Add(size_t n, const point_t &p, const point_t &v);

private:
  struct Record {
    Record(const point_t &p, const point_t &v) : position(p), velocity(v) {}

    point_t position;
    point_t velocity;
  };

  std::vector<Record> particles_;
}; // class ParticleSystem
} // namespace pbf

#endif /* particle_system_h */
