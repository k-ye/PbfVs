#ifndef boundary_constraint_h
#define boundary_constraint_h

#include "typedefs.h"

#include <vector>

namespace pbf {
	struct BoundaryPlane {
		BoundaryPlane(const vec_t& n);

		vec_t position;
		vec_t velocity;
		const vec_t normal;
	};

	class BoundaryConstraintBase {
	public:
		BoundaryConstraintBase() = default;
		virtual ~BoundaryConstraintBase() = default;

		void ApplyBoundaryConstraint();

		void AddBoundary(const BoundaryPlane& bp);

	protected:
		virtual void ApplyAtBoundary_(const BoundaryPlane& bp) = 0;

		std::vector<BoundaryPlane> boundaries_;
	};
}

#endif // boundary_constraint_h