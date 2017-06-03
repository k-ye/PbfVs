#ifndef pbf_solver_gpu_h
#define pbf_solver_gpu_h

#include <thrust\device_vector.h>
#include <thrust\reduce.h>

#include "aabb.h"
#include "pbf_solver_base.h"

namespace pbf {
	template <typename T>
	using d_vector = thrust::device_vector<T>;
	
	float3 Convert(const point_t& pt);
	point_t Convert(const float3& f);

	class CellGridGpu {
	public:
		CellGridGpu(float3 world_sz, float cell_sz);

		float cell_size() const { return cell_sz_; }
		const int3& num_cells_per_dim() const { return num_cells_per_dim_; }
		int total_num_cells() const { return total_num_cells_; }
		
		d_vector<int> cell_is_active_flags;
		d_vector<int> cell_to_active_cell_indices;
		d_vector<int> active_cell_num_ptcs;
		d_vector<int> ptc_begins_in_active_cell;
		d_vector<int> cell_ptc_indices;

		// Below are for debug purpose, we do not need to store these vectors.
		d_vector<int> ptc_to_cell;
		d_vector<int> ptc_offsets_within_cell;
	private:
		float3 world_sz_per_dim_;
		float cell_sz_;
		int3 num_cells_per_dim_;
		int total_num_cells_;
	};

	class GpuParticleNeighbors {
	public:
		// neighbors
		d_vector<int> ptc_num_neighbors;
		d_vector<int> ptc_neighbor_begins;
		d_vector<int> ptc_neighbor_indices;

		inline int* ptc_num_neighbors_ptr() {
			return thrust::raw_pointer_cast(ptc_num_neighbors.data());
		}
		
		inline int* ptc_neighbor_begins_ptr() {
			return thrust::raw_pointer_cast(ptc_neighbor_begins.data());
		}

		inline int* ptc_neighbor_indices_ptr() {
			return thrust::raw_pointer_cast(ptc_neighbor_indices.data());
		}
	};
	
	void UpdateCellGrid(const d_vector<float3>& positions, 
		CellGridGpu* cell_grid);

	void FindParticleNeighbors(const d_vector<float3>& positions,
		const CellGridGpu& cell_grid, const float h, GpuParticleNeighbors* pn);

	class PbfSolverGpu : public PbfSolverBase {
	public:
		PbfSolverGpu() : PbfSolverBase(), num_ptcs_(0) {}
		
		void Update(float dt) override;

	private:
		// overrides		
		void CustomConfigure_(const PbfSolverConfig& config) override;
		
		void CustomInitPs_() override;

		// helpers
		void ResetParticleRecords_();

		void RecordOldPositions_();

		void ApplyGravity_(const float dt);

		void ImposeBoundaryConstraint_();
		
		void FindNeighbors_();
		
		void ComputeLambdas_();
		
		void ComputeDeltaPositions_();
		
		void ApplyDeltaPositions_();

		void UpdateVelocities_(const float dt);
		
		void ComputeVorticities_();
		
		void ComputeVorticityCorrForces_();
		
		void ComputeXsphs_();

		void ApplyVelocityCorrections_(const float dt);
		
		void UpdatePs_();

		inline float3* PositionsPtr_() { return thrust::raw_pointer_cast(d_positions_.data()); }
		inline const float3* PositionsPtr_() const { return thrust::raw_pointer_cast(d_positions_.data()); }
		
		inline float3* VelocitiesPtr_() { return thrust::raw_pointer_cast(d_velocities_.data()); }
		inline const float3* VelocitiesPtr_() const { return thrust::raw_pointer_cast(d_velocities_.data()); }
	
	private:
		float cell_grid_size_;
		int num_ptcs_;

		d_vector<float3> d_positions_;
		d_vector<float3> d_velocities_;
		
		// particle records 
		GpuParticleNeighbors ptc_nb_recs_;
		d_vector<float3> old_positions_;
		d_vector<float> lambdas_;
		d_vector<float3> delta_positions_;
		d_vector<float3> vorticities_;
		d_vector<float3> vorticity_corr_forces_;
		d_vector<float3> xsphs_;
	};

	// For test purpose only
	void Query(const d_vector<float3>& positions, const CellGridGpu& cell_grid,
		const AABB& range, d_vector<int>* cell_num_ptcs_inside);

} // namespace pbf

#endif // pbf_solver_gpu_h
