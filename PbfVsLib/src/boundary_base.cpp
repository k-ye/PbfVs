#include "../include/boundary_base.h"

#include "../include/glm_headers.h"

namespace pbf {
	BoundaryPlane::BoundaryPlane(const vec_t& n)
		: position(0.0f), velocity(0.0f), normal(glm::normalize(n)) {}

	void BoundaryConstraintBase::ApplyBoundaryConstraint() {
		for (const auto& bp : boundaries_) {
			ApplyAtBoundary_(bp);
		}
	}

	void BoundaryConstraintBase::Add(const BoundaryPlane& bp) {
		boundaries_.push_back(bp);
	}

	const BoundaryPlane& BoundaryConstraintBase::Get(size_t i) const {
		return boundaries_[i];
	}
    
    BoundaryPlane* BoundaryConstraintBase::Get(size_t i) { 
        return &boundaries_[i]; 
    }
} // namespace pbf