#ifndef pbf_solver_gpu_h
#define pbf_solver_gpu_h

#include <thrust\device_vector.h>
#include <thrust\reduce.h>
#include "aabb.h"

namespace pbf {
	template <typename T>
	using d_vector = thrust::device_vector<T>;
	
	float3 Convert(const point_t& pt);

	class CellGridGpu {
	public:
		CellGridGpu(float3 world_sz, float cell_sz);

		float cell_size() const { return cell_sz_; }
		const int3& num_cells_per_dim() const { return num_cells_per_dim_; }
		int total_num_cells() const { return total_num_cells_; }
		
		d_vector<int> cell_to_active_cell_indices;
		d_vector<int> active_cell_num_ptcs;
		d_vector<int> ptc_begins_in_active_cell;
		d_vector<int> cell_ptc_indices;

		// Below are for debug purpose, we do not need to store these vectors.
		d_vector<int> ptc_to_cell;
		d_vector<int> cell_is_active_flags;
		d_vector<int> ptc_offsets_within_cell;
	private:
		float3 world_sz_per_dim_;
		float cell_sz_;
		int3 num_cells_per_dim_;
		int total_num_cells_;
	};

	class ParticleNeighbors {
	public:
		d_vector<int> ptc_num_neighbors;
		d_vector<int> ptc_neighbor_begins;
		d_vector<int> ptc_neighbor_indices;
	};

	void UpdateCellGrid(const d_vector<float3>& positions, 
		CellGridGpu* cell_grid);

	void FindParticleNeighbors(const d_vector<float3>& positions,
		const CellGridGpu& cell_grid, const float cell_sz, const int3& num_cells_dim,
		const float h, ParticleNeighbors* pn);

	// For test purpose only
	void Query(const d_vector<float3>& positions, const CellGridGpu& cell_grid,
		const AABB& range, d_vector<int>* cell_num_ptcs_inside);
} // namespace pbf

#endif // pbf_solver_gpu_h
