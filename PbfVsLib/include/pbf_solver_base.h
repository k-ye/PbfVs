#ifndef pbf_solver_base_h 
#define pbf_solver_base_h 

#include "basic.h"
#include "kernel.h"
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

		float world_size;
		float spatial_hash_cell_size;
	};

	class PbfSolverBase {
	public:
		PbfSolverBase();

		void Configure(const PbfSolverConfig& config);
		
		void InitParticleSystems(ParticleSystem* ps);

		virtual void Update(float dt) = 0;
	
	protected:
		// Called as the last step in Configure().
		virtual void CustomConfigure_(const PbfSolverConfig& config) {}
		// Called as the last step in InitParticleSystems, at which
		// point ps_ is guaranteed to be initialized.
		virtual void CustomInitPs_() {}
		
		void ResetParticleRecords_();
		
		void RecordOldPositions_();
	
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

		float world_size_;

		WKernel kernel_{};
		ParticleSystem* ps_;

		class ParticleRecord
		{
		public:
			void ClearNeighbors() { neighbor_idxs.clear(); }

			void AddNeighbor(size_t i) { neighbor_idxs.insert(i); }

		public:
			// std::vector<size_t> neighbor_idxs;
			std::unordered_set<size_t> neighbor_idxs;
			float lambda{ 0.0f };

			vec_t old_pos{ 0.0f };
			vec_t delta_pos{ 0.0f };
			vec_t vorticity{ 0.0f };
		};
		
		std::vector<ParticleRecord> ptc_records_;
	};
} // namespace pbf

#endif // pbf_solver_base_h 