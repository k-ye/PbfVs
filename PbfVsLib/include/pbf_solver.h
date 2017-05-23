#ifndef pbf_solver_h
#define pbf_solver_h

#include <vector>

#include "gravity.h"
#include "pbf_solver_base.h"
#include "sh_position_getter.h"
#include "spatial_hash.h"

namespace pbf
{
	class PbfSolver : public PbfSolverBase {
	public:
		PbfSolver() : PbfSolverBase() {}

		void Update(float dt) override;
	private:
		// overrides
		void CustomConfigure_(const PbfSolverConfig& config) override;
		
		void CustomInitPs_() override;

		// helpers to implement this solver
		void ImposeBoundaryConstraint_();

		void FindNeighbors_();

		float ComputeLambda_(size_t p_i) const;

		// @p_i: index of particle i.
		float ComputeDensityConstraint_(size_t p_i) const;

		vec_t ComputeDeltaPos_(size_t p_i) const;

		float ComputScorr_(const vec_t vec_ji) const;

		vec_t ComputeVorticityCorrForce_(size_t p_i) const;

		vec_t ComputeVorticity_(size_t p_i) const;
		
		vec_t PbfSolver::ComputeEta_(size_t p_i) const;

		vec_t ComputeXsph_(size_t p_i) const;
	private:
		GravityEffect gravity_{};
		SpatialHash<size_t, PositionGetter> spatial_hash_;
	};
}
#endif /* pbf_solver_h */
