#include "../include/boundary_base.h"

#include "../include/glm_headers.h"

namespace pbf {
	BoundaryPlane::BoundaryPlane(const vec_t& n)
		: position(0.0f), velocity(0.0f), normal(glm::normalize(n)) {}

	void BoundaryConstraintBase::ApplyBoundaryConstraint() {

	}

	void BoundaryConstraintBase::AddBoundary(const BoundaryPlane& bp) {
		boundaries_.push_back(bp);
	}
} // namespace pbf