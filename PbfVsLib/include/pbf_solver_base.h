#ifndef pbf_solver_base_h 
#define pbf_solver_base_h 

#include "typedefs.h"
#include "particle_system.h"

#include <cmath>
#include <unordered_set>
#include <vector>

namespace pbf {
	struct PbfSolverConfig {
		float h;
		float mass;
		float rho_0;
		float epsilon;
		unsigned num_iters;

		float corr_delta_q_coeff;
		float corr_k;
		unsigned corr_n;

		float vorticity_epsilon;
		float xsph_c;

        float world_size_x;
        float world_size_y;
        float world_size_z;
        // TODO: spatial_hash should be renamed
		float spatial_hash_cell_size;
	};

	class PbfSolverBase {
	public:
		PbfSolverBase();
		virtual ~PbfSolverBase() = default;

		void Configure(const PbfSolverConfig& config);
		
		void InitParticleSystems(ParticleSystem* ps);

		virtual void Update(float dt) = 0;
	
	protected:
		// Called as the last step in Configure().
		virtual void CustomConfigure_(const PbfSolverConfig& config) {}
		// Called as the last step in InitParticleSystems, at which
		// point ps_ is guaranteed to be initialized.
		virtual void CustomInitPs_() {}
	
	protected:
		// kernel function h
		float h_;
		// Mass of a particle. All particles have the same mass.
		float mass_;
		// Rest density of a particle.
		float rho_0_;
		// Reciprocal of rho_0_;
		float rho_0_recpr_;
		// Epsilon in Eq (11)
		float epsilon_;
		unsigned num_iters_;

		// Tensile instanbility correction
		float corr_delta_q_coeff_;
		float corr_k_;
		unsigned corr_n_;
		
		float vorticity_epsilon_;
		float xsph_c_;

        float world_size_x_;
        float world_size_y_;
        float world_size_z_;


		ParticleSystem* ps_;
	};
} // namespace pbf

#endif // pbf_solver_base_h 