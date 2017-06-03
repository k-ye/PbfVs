#include "../include/pbf_solver_base.h"

namespace pbf {
	PbfSolverBase::PbfSolverBase()
		: h_(0.0f), mass_(0.0f), rho_0_(0.0f), rho_0_recpr_(0.0f)
		, epsilon_(0.0f), num_iters_(0), corr_delta_q_coeff_(0.0f)
		, corr_k_(0.0f) , corr_n_(0.0f), vorticity_epsilon_(0.0f)
		, xsph_c_(0.0f), world_size_(0.0f), ps_(nullptr) {}
		
	void PbfSolverBase::Configure(const PbfSolverConfig& config) {
		// init consts for the solver
		h_ = config.h;
		mass_ = config.mass;
		rho_0_ = config.rho_0;
		rho_0_recpr_ = 1.0f / rho_0_;
		epsilon_ = config.epsilon;
		num_iters_ = config.num_iters;
		corr_delta_q_coeff_ = config.corr_delta_q_coeff;
		corr_k_ = config.corr_k;
		corr_n_ = config.corr_n;
		vorticity_epsilon_ = config.vorticity_epsilon;
		xsph_c_ = config.xsph_c;

		world_size_ = config.world_size;
		// Additional configuration by the subclass
		CustomConfigure_(config);
	}
	
	void PbfSolverBase::InitParticleSystems(ParticleSystem* ps) {
		ps_ = ps;
		// Additional particle system initialization by the subclass
		CustomInitPs_();
	}
} // namespace pbf