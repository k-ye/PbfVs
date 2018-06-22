#include "../include/boundary_gpu.h"

#include "../include/cuda_basic.h"
#include "../include/shared_math.h"

namespace pbf {
namespace impl_ {
__global__ static void
ApplyBoundaryConstraintKernel(const float3 boundary_pos,
                              const float3 boundary_vel,
                              const float3 boundary_normal, const int num_ptcs,
                              float3 *positions, float3 *velocities) {
  // precondition: |boundary_normal| is normalized, length is 1.
  const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (ptc_i >= num_ptcs)
    return;

  float3 pos = positions[ptc_i];
  float3 vel = velocities[ptc_i];
  // This is actually not the distance because we keep the sign
  const float plane_to_ptc_dist = dot(pos - boundary_pos, boundary_normal);
  if (plane_to_ptc_dist <= kFloatEpsilon) {
    // Entering this branch means the particle is colliding with/already on the
    // "out" side of the boundary plane. We need to correct particle position
    // and (probably) its velocity.
    //
    // this is the projected point of the particle on the boundary plane
    const float3 proj_pos = pos - (plane_to_ptc_dist * boundary_normal);
    pos = proj_pos;

    // make sure |boundary_vel| is large enough to make the computation stable.
    const float boundary_vel_dot = dot(boundary_vel, boundary_vel);
    float3 proj_vel;
    if (boundary_vel_dot <= kFloatEpsilon) {
      // |boundary_vel| is almost zero, it is not stable to project particle's
      // velocity along the boundary's. We project onto the boundary's normal
      // instead.
      proj_vel = dot(vel, boundary_normal) * boundary_normal;
    } else {
      // computationally stable
      proj_vel = dot(vel, boundary_vel) * boundary_vel / boundary_vel_dot;
    }
    // original particle velocity can be decompoosed into two components:
    // projected vel and perpendicular vel.
    const float3 proj_vel_diff = proj_vel - boundary_vel;
    if (dot(proj_vel_diff, boundary_normal) <= kFloatEpsilon) {
      const float3 perp_vel = vel - proj_vel;
      // particle's projected velocity needs to be aligned with the boundary's
      // in terms of the strength
      vel = boundary_vel + perp_vel;
    }
  }
  positions[ptc_i] = pos;
  velocities[ptc_i] = vel;
}
} // namespace impl_

void BoundaryConstraintGpu::SetPsAdaptor(
    std::shared_ptr<ParticleSystemGpuAdaptor> pa) {
  ps_adaptor_ = pa;
}

void BoundaryConstraintGpu::ApplyAtBoundary_(const BoundaryPlane &bp) {
  const float3 boundary_pos = Convert(bp.position);
  const float3 boundary_vel = Convert(bp.velocity);
  const float3 boundary_normal = Convert(bp.normal);

  const int num_ptcs = ps_adaptor_->NumParticles();
  const int num_blocks_ptc =
      ((num_ptcs + kNumThreadPerBlock - 1) / kNumThreadPerBlock);

  impl_::ApplyBoundaryConstraintKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
      boundary_pos, boundary_vel, boundary_normal, num_ptcs,
      ps_adaptor_->PositionsPtr(), ps_adaptor_->VelocitiesPtr());
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
} // namespace pbf
