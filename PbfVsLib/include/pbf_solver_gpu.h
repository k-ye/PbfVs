#ifndef pbf_solver_gpu_h
#define pbf_solver_gpu_h

#include <thrust\device_vector.h>
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

	private:
		float3 world_sz_per_dim_;
		float cell_sz_;
		int3 num_cells_per_dim_;
		int total_num_cells_;
	};

	void UpdateCellGrid(const d_vector<float3>& positions, 
		CellGridGpu* cell_grid);

	void Query(const d_vector<float3>& positions, const CellGridGpu& cell_grid,
		const AABB& range, d_vector<int>* cell_num_ptcs_inside);
} // namespace pbf

#endif // pbf_solver_gpu_h
