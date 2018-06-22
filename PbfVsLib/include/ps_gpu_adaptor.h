#ifndef ps_gpu_adaptor_h
#define ps_gpu_adaptor_h

#include "particle_system.h"
#include "typedefs.h"
#include <thrust/device_vector.h>

namespace pbf {
class ParticleSystemGpuAdaptor {
private:
  typedef thrust::device_vector<float3> vector_t;

public:
  ParticleSystemGpuAdaptor() = default;
  // non copy/assign-able
  ParticleSystemGpuAdaptor(const ParticleSystemGpuAdaptor &) = delete;
  ParticleSystemGpuAdaptor &
  operator=(const ParticleSystemGpuAdaptor &) = delete;

  ParticleSystem *ps();
  size_t NumParticles() const;

  void SetPs(ParticleSystem *ps);

  void UpdatePs();

  float3 *PositionsPtr();
  float3 *VelocitiesPtr();

  vector_t *PositionsVec();
  const vector_t &PositionsVec() const;

  vector_t *VelocitiesVec();
  const vector_t &VelocitiesVec() const;

private:
  void Precondition_() const;

  ParticleSystem *ps_{nullptr};
  bool inited_{false};
  vector_t d_positions_;
  vector_t d_velocities_;
};
} // namespace pbf

#endif // ps_gpu_adaptor_h
