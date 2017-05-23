#include "../include/pbf_solver.h"

#include <algorithm>
#include <cmath>

#include "../include/aabb.h"
#include "../include/basic.h"

namespace pbf {

	void PbfSolver::CustomConfigure_(const PbfSolverConfig& config) {
		// init spatial hash
		spatial_hash_.set_cell_size(config.spatial_hash_cell_size);
	}
	
	void PbfSolver::CustomInitPs_() {
		// First reset all the auxiliary data structures
		spatial_hash_.Clear();
		// Then construct them based on the new ps instance
		PositionGetter pg{ ps_ };
		spatial_hash_.set_pos_getter(pg);
		for (size_t ptc_i = 0; ptc_i < ps_->NumParticles(); ++ptc_i) {
			spatial_hash_.Add(ptc_i);
		}
	}
	
	void PbfSolver::Update(float dt) {
		ResetParticleRecords_();
		RecordOldPositions_();

		// apply the gravity and the boundary constraint
		gravity_.Evaluate(dt, ps_);
		ImposeBoundaryConstraint_();

		// update the particle positions in the spatial hash
		// and find the neighbors for each particle.
		spatial_hash_.UpdateAll();
		FindNeighbors_();

		for (unsigned itr = 0; itr < num_iters_; ++itr) {
			// Compute lambda
			for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
				auto& ptc_rec_i = ptc_records_[p_i];
				ptc_rec_i.lambda = ComputeLambda_(p_i);
			}
			// Compute delta position 
			for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
				auto& ptc_rec_i = ptc_records_[p_i];
				ptc_rec_i.delta_pos = ComputeDeltaPos_(p_i);
			}
			// Update particle position
			for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
				auto ptc_i = ps_->Get(p_i);
				auto new_pos_i = ptc_i.position() + ptc_records_[p_i].delta_pos;
				ptc_i.set_position(new_pos_i);
			}
		}

		// Update velocity
		ImposeBoundaryConstraint_();
		for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
			auto ptc_i = ps_->Get(p_i);
			const auto old_pos_i = ptc_records_[p_i].old_pos;
			const auto new_pos_i = ptc_i.position();
			const auto new_vel_i = (new_pos_i - old_pos_i) / dt;
			ptc_i.set_velocity(new_vel_i);
		}
		// Compute Vorticity
		for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
			auto& ptc_rec_i = ptc_records_[p_i];
			ptc_rec_i.vorticity = ComputeVorticity_(p_i);
		}
		// Apply vorticity confinement and XSPH
		for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
			auto ptc_i = ps_->Get(p_i);
			const vec_t vel_i = ptc_i.velocity();
			vec_t new_vel_i = vel_i;
			new_vel_i += (ComputeVorticityCorrForce_(p_i) * dt);
			new_vel_i += ComputeXsph_(p_i);
			ptc_i.set_velocity(new_vel_i);
		}
	}
	
	void PbfSolver::ImposeBoundaryConstraint_() {
		for (size_t ptc_i = 0; ptc_i < ps_->NumParticles(); ++ptc_i) {
			auto ptc = ps_->Get(ptc_i);
			auto pos = ptc.position();
			auto vel = ptc.velocity();

			for (int c = 0; c < 3; ++c) {
				if (pos[c] <= 0.0f || (pos[c] >= world_size_ - kFloatEpsilon)) {
					vel[c] = 0.0f;
					pos[c] = std::max(0.0f, std::min(world_size_ - kFloatEpsilon, pos[c]));
				}
			}
			
			ptc.set_position(pos);
			ptc.set_velocity(vel);
		}
	}

	void PbfSolver::FindNeighbors_() {
		assert(ps_->NumParticles() == spatial_hash_.size());
		
		const float half_h = h_ * 0.5f;
		const float h_sqr = h_ * h_;
		for (size_t sh_i = 0; sh_i < spatial_hash_.size(); ++sh_i) {
			size_t p_i = spatial_hash_.Get(sh_i);
			auto ptc_i = ps_->Get(p_i);
			auto& ptc_rec_i = ptc_records_[p_i];
			ptc_rec_i.ClearNeighbors();
			
			const AABB query_aabb{ ptc_i.position(), half_h };
			auto query_result = spatial_hash_.Query(query_aabb);
			for (size_t sh_j : query_result) {
				if (sh_i == sh_j) {
					continue;
				}
				size_t p_j = spatial_hash_.Get(sh_j);
				auto ptc_j = ps_->Get(p_j);

				vec_t pos_diff_ji = ptc_j.position() - ptc_i.position();
				float dist_sqr = glm::dot(pos_diff_ji, pos_diff_ji);
				if (dist_sqr <= h_sqr) {
					ptc_rec_i.AddNeighbor(p_j);
				}
			}
		}
	}

	float PbfSolver::ComputeLambda_(size_t p_i) const {
		// Eq (8) (11)
		const auto pos_i = ps_->Get(p_i).position();
		
		vec_t gradient_i{ 0.0f };
		float sum_gradient = 0.0f;
		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_j = ps_->Get(p_j).position();

			vec_t gradient_j = kernel_.Gradient(pos_i - pos_j) * rho_0_recpr_;
			sum_gradient += glm::dot(gradient_j, gradient_j);
			gradient_i += gradient_j;
		}
		sum_gradient += glm::dot(gradient_i, gradient_i);

		float density_constraint = ComputeDensityConstraint_(p_i);
		float result = (-density_constraint) / (sum_gradient + epsilon_);
		return result;
	}

	// @p_i: index of particle i.
	float PbfSolver::ComputeDensityConstraint_(size_t p_i) const {
		// Eq (1)
		float result = 0.0f;
		const auto pos_i = ps_->Get(p_i).position();

		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_j = ps_->Get(p_j).position();
			
			result += mass_ * kernel_.Evaluate(pos_i - pos_j);
		}
		result = (result * rho_0_recpr_) - 1.0f;
		return result;
	}

	vec_t PbfSolver::ComputeDeltaPos_(size_t p_i) const {
		// Eq (12), (14)
		const auto pos_i = ps_->Get(p_i).position();
		const float lambda_i = ptc_records_[p_i].lambda;
		vec_t result{ 0.0f };

		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_j = ps_->Get(p_j).position();
			const float lambda_j = ptc_records_[p_j].lambda;
			const auto pos_diff_ji = pos_i - pos_j;
			const float scorr_ij = ComputScorr_(pos_diff_ji);
			result += (lambda_i + lambda_j + scorr_ij) * kernel_.Gradient(pos_diff_ji);
		}
		result *= rho_0_recpr_;
		return result;
	}
		
	float PbfSolver::ComputScorr_(const vec_t pos_diff_ji) const {
		// Eq (13)
		float x = kernel_.Evaluate(pos_diff_ji) / kernel_.Evaluate(corr_delta_q_coeff_);
		float result = (-corr_k_) * std::pow(x, (float)corr_n_);
		return result;
	}
		
	vec_t PbfSolver::ComputeVorticityCorrForce_(size_t p_i) const {
		// Eq (16)
		vec_t eta = ComputeEta_(p_i);
		float eta_len = glm::length(eta);
		if (eta_len <= kFloatEpsilon)
			return vec_t{ 0.0f };
		eta = glm::normalize(eta);
		const auto omega_i = ptc_records_[p_i].vorticity;
		vec_t result = (vorticity_epsilon_ * glm::cross(eta, omega_i));
		return result;
	}

	vec_t PbfSolver::ComputeVorticity_(size_t p_i) const {
		// Eq (15)
		vec_t result{ 0.0f };
		const auto pos_i = ps_->Get(p_i).position();
		const auto vel_i = ps_->Get(p_i).velocity();
		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_j = ps_->Get(p_j).position();
			const auto vel_j = ps_->Get(p_j).velocity();
			
			const vec_t vel_diff_ij = vel_j - vel_i;
			// const vec_t gradient = kernel_.WViscosity(pos_i - pos_j);
			const vec_t gradient = kernel_.Gradient(pos_i - pos_j);
			result += glm::cross(vel_diff_ij, gradient);
		}
		return result;
	}

	vec_t PbfSolver::ComputeEta_(size_t p_i) const {
		// eta = grad|omega_i|
		vec_t result{ 0.0f };
		const auto pos_i = ps_->Get(p_i).position();
		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_diff_ji = pos_i - ps_->Get(p_j).position();
			const float omega_j_len = glm::length(ptc_records_[p_j].vorticity);
			const auto gradient = kernel_.Gradient(pos_diff_ji);
			result += (omega_j_len * gradient);
		}
		return result;
	}
	
	vec_t PbfSolver::ComputeXsph_(size_t p_i) const {
		// Eq (17)
		vec_t result{ 0.0f };
		const auto pos_i = ps_->Get(p_i).position();
		const auto vel_i = ps_->Get(p_i).velocity();

		for (size_t p_j : ptc_records_[p_i].neighbor_idxs) {
			const auto pos_j = ps_->Get(p_j).position();
			const auto vel_j = ps_->Get(p_j).velocity();

			const vec_t vel_diff_ij = vel_j - vel_i;
			float w = kernel_.Evaluate(pos_i - pos_j);
			result += (w * vel_diff_ij);
		}
		result *= xsph_c_;
		return result;
	}
} // namespace pbf