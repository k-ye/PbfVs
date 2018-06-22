#include "../include/ps_gpu_adaptor.h"

#include <cassert>
#include <thrust/host_vector.h>

namespace pbf {
void ParticleSystemGpuAdaptor::Precondition_() const {
  assert((ps_ != nullptr) && inited_);
}

ParticleSystem *ParticleSystemGpuAdaptor::ps() { return ps_; }

size_t ParticleSystemGpuAdaptor::NumParticles() const {
  Precondition_();
  return ps_->NumParticles();
}

void ParticleSystemGpuAdaptor::SetPs(ParticleSystem *ps) {
  assert(ps != nullptr);
  ps_ = ps;
  const size_t num_ptcs = ps_->NumParticles();

  d_positions_.resize(num_ptcs);
  d_velocities_.resize(num_ptcs);
  for (size_t i = 0; i < num_ptcs; ++i) {
    const auto ptc_i = ps_->Get(i);
    d_positions_[i] = Convert(ptc_i.position());
    d_velocities_[i] = Convert(ptc_i.velocity());
  }

  inited_ = true;
}

void ParticleSystemGpuAdaptor::UpdatePs() {
  Precondition_();
  using thrust::host_vector;
  host_vector<float3> h_positions{d_positions_};
  host_vector<float3> h_velocities{d_velocities_};

  for (size_t p_i = 0; p_i < NumParticles(); ++p_i) {
    auto ptc_i = ps_->Get(p_i);
    ptc_i.set_position(Convert(h_positions[p_i]));
    ptc_i.set_velocity(Convert(h_velocities[p_i]));
  }
}

float3 *ParticleSystemGpuAdaptor::PositionsPtr() {
  Precondition_();
  return thrust::raw_pointer_cast(d_positions_.data());
}

float3 *ParticleSystemGpuAdaptor::VelocitiesPtr() {
  Precondition_();
  return thrust::raw_pointer_cast(d_velocities_.data());
}

ParticleSystemGpuAdaptor::vector_t *ParticleSystemGpuAdaptor::PositionsVec() {
  Precondition_();
  return &d_positions_;
}

const ParticleSystemGpuAdaptor::vector_t &
ParticleSystemGpuAdaptor::PositionsVec() const {
  Precondition_();
  return d_positions_;
}

ParticleSystemGpuAdaptor::vector_t *ParticleSystemGpuAdaptor::VelocitiesVec() {
  Precondition_();
  return &d_velocities_;
}

const ParticleSystemGpuAdaptor::vector_t &
ParticleSystemGpuAdaptor::VelocitiesVec() const {
  Precondition_();
  return d_velocities_;
}
} // namespace pbf
