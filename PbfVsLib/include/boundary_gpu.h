#ifndef boundary_gpu_h
#define boundary_gpu_h

#include "boundary_base.h"
#include "ps_gpu_adaptor.h"

#include <memory>	// std::shared_ptr

namespace pbf {
    class BoundaryConstraintGpu : public BoundaryConstraintBase {
    public:
        BoundaryConstraintGpu() = default;
        BoundaryConstraintGpu(const BoundaryConstraintGpu&) = default;
        BoundaryConstraintGpu& operator=(const BoundaryConstraintGpu&) = default;

        void SetPsAdaptor(std::shared_ptr<ParticleSystemGpuAdaptor> pa);

    private:
        void ApplyAtBoundary_(const BoundaryPlane& bp) override;

    private:
        std::shared_ptr<ParticleSystemGpuAdaptor> ps_adaptor_{ nullptr };
    };
} // namespace pbf

#endif // boundary_gpu_h