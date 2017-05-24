#include "../include/pbf_solver_gpu.h"

// CUDA
#include "cuda_runtime.h"
#include <thrust\device_vector.h>

namespace pbf {
namespace impl_ {
	// ParticleSystemGpu
	//
	// CellGridGpu (equivalent to SpatialHash on CPU)
	// - Need to config the grid size and cell size before usage. Once
	//   configured, they are not allowed to be modified. Not adaptive.
	// - active cell: a cell that contains at least one particle.
	//
	// Example: a particle system of 8 particles, a cell grid of 5 cells.
	// We will illustrate the necessary arrays for updating the cell grid.
	//
	// | 0 | 1 | 2 | 3 | 4 |  cell index 
	// cell_num_ptcs
	//   3   0   1   4   0
	// - size: #cells
	// - number of particles in each cell, including inactive(empty) ones
	// - sum of this array is the total number of particles
	//
	// cell_is_active_flags
	//   1   0   1   1   0
	// - size: #cells
	// - sum of this array is the number of active cells
	//
	// cell_to_active_cell_indices
	//   0   1   1   2   3
	// - size: #cells
	// - a prefix scan of |cell_is_active_flags|
	//
	// active_cell_num_ptcs
	//   3   1   4
	// - size: #active cells
	// - a compact of |cell_num_ptcs| accoording to |cell_to_active_cell_indices|
	// - sum of this array is the total number of particles
	//
	// ptc_begins_in_active_cell 
	//   0   3   4
	// - size: #active cells
	// - beginning index of the particle in each (active) cell in 
	//   |cell_ptc_indices|
	// - a prefix scan of |active_cell_num_ptcs|
	//
	// cell_ptc_indices
	// - size: #ptcs
	// - each slot stores a particle index in the particle system. these particle
	//   indices are arranged in a way that particles within the same grid cell
	//   are continuously stored inside |cell_ptc_indices|.
	//
	// ptc_offsets_within_cell
	// - size: #particles
	// - for any given particle index, |p_i|, we can get its position, |pos_i|,
	//   and its cell, |cell_i|. Then:
	//	 // the active cell index of |cell_i| in which |ptc_i| lives
	//	 ac_idx = cell_to_active_cell_indices[cell_i];
	//   // the beginning index of the particles within |cell_i|
	//	 // in |cell_ptc_indices|
	//   ptc_begin_idx = ptc_begins_in_active_cell[ac_index];
	//   p_i' = ptc_begin_idx + ptc_offset_within_cell[p_i]; 
	//   assert(p_i == p_i');
	//
	// Find neighbors for each particle:
	// ptc_num_neighbors
	// - size: #particles
	// - stores the number of neighbor particles for each particle
	//
	// ptc_neighbor_begins
	// - size: #particles
	// - ~[p_i] stores the beginning index of particle |p_i|'s neighbors
	//   in |ptc_neighbor_indices|
	//
	// ptc_neighbor_indices
	// - size: sum of |ptc_num_neighbors| 
	
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

	void ResetNumPtcsInCell(thrust::device_vector<int>* cell_num_ptcs) {
		const size_t sz = cell_num_ptcs->size();
		cell_num_ptcs->assign(sz, 0);
	}

	__global__ void CountNumPtcsInCell(const float3* positions, 
		const int num_ptcs, const float cell_sz,
		const int3 num_cells_dim, int* cell_num_ptcs,
		int* ptc_offset_within_cell) {
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		int3 ptc_cell = GetCell(positions[ptc_i], cell_sz);
		int cell_index = GetCellIndex(ptc_cell, num_cells_dim);
		int offs = atomicAdd(&cell_num_ptcs[cell_index], 1);
		ptc_offset_within_cell[ptc_i] = offs;
	}

} // namespace impl_
} // namespace pbf