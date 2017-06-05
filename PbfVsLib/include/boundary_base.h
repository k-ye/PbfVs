#ifndef boundary_constraint_h
#define boundary_constraint_h

#include "typedefs.h"

#include <vector>

namespace pbf {
	struct BoundaryPlane {
        BoundaryPlane(const vec_t& n = vec_t{ 0.0f });
        BoundaryPlane(const BoundaryPlane&) = default;
        BoundaryPlane& operator=(const BoundaryPlane&) = default;

        vec_t position{ 0.0f };
        vec_t velocity{ 0.0f };
        vec_t normal;
	};

	class BoundaryConstraintBase {
	public:
		BoundaryConstraintBase() = default;
		virtual ~BoundaryConstraintBase() = default;

		void ApplyBoundaryConstraint();

		void Add(const BoundaryPlane& bp);

		const BoundaryPlane& Get(size_t i) const;
        BoundaryPlane* Get(size_t i);
	
	protected:
		virtual void ApplyAtBoundary_(const BoundaryPlane& bp) = 0;

		std::vector<BoundaryPlane> boundaries_;
	};
}

#endif // boundary_constraint_h