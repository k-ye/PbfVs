#include "../include/pbf_solver_gpu.h"

// CUDA
#include "cuda_runtime.h"
#include <thrust\device_vector.h>

namespace pbf {
namespace impl_ {
	// ParticleSystemGpu
	// CellGridGpu (equivalent to SpatialHash on CPU)
	// - Need to config the grid size and cell size before usage. Once
	//   configured, they are not allowed to be modified. Not adaptive.
	// - active cell: a cell that contains at least one particle
	__device__ int3 GetCell(float3 pos, float cell_sz) {
		const float cell_sz_recpr = 1.0f / cell_sz;
		int cx = (int)(pos.x * cell_sz_recpr);
		int cy = (int)(pos.y * cell_sz_recpr);
		int cz = (int)(pos.z * cell_sz_recpr);
		return make_int3(cx, cy, cz);
	}

	__device__ int GetCellIndex(int3 cell, int3 num_cells_dim) {
		int result = cell.y * num_cells_dim.z;
		result = (result + cell.z) * num_cells_dim.x;
		result += cell.x;
		return result;
	}

	__device__ bool IsValidCell(int3 cell, int3 num_cells_dim) {
		return ((0 <= cell.x && cell.x < num_cells_dim.x) &&
			(0 <= cell.y && cell.y < num_cells_dim.y) &&
			(0 <= cell.z && cell.z < num_cells_dim.z));
	}

	void ResetNumPtcsInCell(thrust::device_vector<int>* num_ptcs_in_cell) {
		const size_t sz = num_ptcs_in_cell->size();
		num_ptcs_in_cell->assign(sz, 0);
	}

	__global__ void CountNumPtcsInCell(const float3* positions, 
		const int num_ptcs, const float cell_sz,
		const int3 num_cells_dim, int* num_ptcs_in_cell,
		int* ptc_offset_within_cell) {
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		int3 ptc_cell = GetCell(positions[ptc_i], cell_sz);
		int cell_index = GetCellIndex(ptc_cell, num_cells_dim);
		int offs = atomicAdd(&d_num_ptcs_in_cell[cell_index], 1);
		ptc_offset_within_cell[ptc_i] = offs;
	}

} // namespace impl_
} // namespace pbf