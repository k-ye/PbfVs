#include "../include/pbf_solver_gpu.h"

#include "../include/kernel.cuh"

// CUDA
#include "../include/cuda_basic.h"
#include "../include/helper_math.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

namespace pbf {
	constexpr int kNumThreadPerBlock = 512;
	
	float3 Convert(const point_t& pt) { return make_float3(pt.x, pt.y, pt.z); }

	point_t Convert(const float3& f) { return point_t{ f.x, f.y, f.z }; }

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
	// cell index
	// | 0 | 1 | 2 | 3 | 4 |   
	//
	// ptc_to_cell
	// - size: #particles
	// - a map between each particle index to its cell 
	//   (NOT the active cell) index
	//
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
	//   // the active cell index of |cell_i| in which |ptc_i| lives
	//   ac_idx = cell_to_active_cell_indices[cell_i];
	//   // the beginning index of the particles within |cell_i|
	//   // in |cell_ptc_indices|
	//   ptc_begin_idx = ptc_begins_in_active_cell[ac_idx];
	//   p_i' = cell_ptc_indices[ptc_begin_idx + ptc_offset_within_cell[p_i]]; 
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

	int ComputeNumBlocks(int num) {
		return ((num + kNumThreadPerBlock - 1) / kNumThreadPerBlock);
	}

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

	__device__ bool IsCellInRange(int3 cell, int3 num_cells_dim) {
		return ((0 <= cell.x && cell.x < num_cells_dim.x) &&
			(0 <= cell.y && cell.y < num_cells_dim.y) &&
			(0 <= cell.z && cell.z < num_cells_dim.z));
	}

	__device__ float DistanceSquare(float3 a, float3 b) {
		float x = a.x - b.x;
		float y = a.y - b.y;
		float z = a.z - b.z;
		float result = x * x + y * y + z * z;
		return result;
	}
	
	__device__ bool IsInside(const float3& pt, const 
		float3& min, const float3& max) {
        bool cond = (min.x <= pt.x) && (pt.x <= max.x) &&
        (min.y <= pt.y) && (pt.y <= max.y) &&
        (min.z <= pt.z) && (pt.z <= max.z);
        return cond;
	}
	
	/////
	// CellGrid
	/////

	// - compute |ptc_to_cell| 
	// - count |cell_num_ptcs|
	// - set the offset of each partilce in |ptc_offset_within_cell|.
	__global__ void CellGridEntryPointKernel(const float3* positions,
		const int num_ptcs, const float cell_sz, const int3 num_cells_dim,
		int* ptc_to_cell, int* cell_num_ptcs , int* ptc_offsets_within_cell) 
	{
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		int3 ptc_cell = GetCell(positions[ptc_i], cell_sz);
		int cell_index = GetCellIndex(ptc_cell, num_cells_dim);
		
		ptc_to_cell[ptc_i] = cell_index;
		// Count the number of particles in |ptc_cell|. The returned
		// value is also used as this particle's unique offset.
		int offs = atomicAdd(&cell_num_ptcs[cell_index], 1);
		(void)offs;
		ptc_offsets_within_cell[ptc_i] = offs;
	}

	// set |cell_is_active_flags|
	__global__ void SetCellIsActiveFlagsKernel(const int* cell_num_ptcs,
		const int num_cells, int* cell_is_active_flags) 
	{
		const int cell_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (cell_i >= num_cells) return;
		cell_is_active_flags[cell_i] = (cell_num_ptcs[cell_i] > 0);
	}

	// compute |cell_to_active_cell_indices|
	void ComputeCellToActiveCellIndices(
		const d_vector<int>& cell_is_active_flags,
		d_vector<int>* cell_to_active_cell_indices) 
	{
		assert(cell_is_active_flags.size() == 
			cell_to_active_cell_indices->size());
		thrust::exclusive_scan(thrust::device, 
			cell_is_active_flags.begin(), cell_is_active_flags.end(),
			cell_to_active_cell_indices->begin(), 0);
	}

	__global__ void CompactKernel(const int* input, const int* flag, 
		const int* compact_indices, const int size, int* output) 
	{
		const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (idx >= size) return;
		if (flag[idx] != 0) {
			const int compact_idx = compact_indices[idx];
			output[compact_idx] = input[idx];
		}
	}

	// compact |cell_num_ptcs| to get |active_cell_num_ptcs|
	void ComputeActiveCellNumPtcs(const d_vector<int>& cell_num_ptcs,
		const d_vector<int>& cell_is_active_flags,
		const d_vector<int>& cell_to_active_cell_indices,
		d_vector<int>* active_cell_num_ptcs)
	{
		const int size = cell_is_active_flags.size();
		const int num_blocks = ComputeNumBlocks(size);
		const int* input = thrust::raw_pointer_cast(cell_num_ptcs.data());
		const int* flags = thrust::raw_pointer_cast(cell_is_active_flags.data());
		const int* compact_indices = thrust::raw_pointer_cast(
			cell_to_active_cell_indices.data());
		int* output = thrust::raw_pointer_cast(active_cell_num_ptcs->data());
		CompactKernel<<<num_blocks, kNumThreadPerBlock>>> (
			input, flags, compact_indices, size, output);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	// compute |ptc_begins_in_active_cell|
	void ComputePtcBeginsInActiveCell(
		const d_vector<int>& active_cell_num_ptcs,
		d_vector<int>* ptc_begins_in_active_cell)
	{
		assert(active_cell_num_ptcs.size() == 
			ptc_begins_in_active_cell->size());
		thrust::exclusive_scan(thrust::device, 
			active_cell_num_ptcs.begin(), active_cell_num_ptcs.end(),
			ptc_begins_in_active_cell->begin(), 0);
	}

	// compute |cell_ptc_indices|
	__global__ void ComputeCellPtcIndicesKernel(
		const int* ptc_to_cell, const int* cell_to_active_cell_indices, 
		const int* ptc_begins_in_active_cell, 
		const int* ptc_offsets_within_cell,
		const int num_ptcs, int* cell_ptc_indices)
	{
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		const int cell_i = ptc_to_cell[ptc_i];
		// active cell index
		const int ac_idx = cell_to_active_cell_indices[cell_i];
		const int ptc_begin_index = ptc_begins_in_active_cell[ac_idx];
		const int i = ptc_begin_index + ptc_offsets_within_cell[ptc_i];
		cell_ptc_indices[i] = ptc_i;
	}

	/////
	// Find Neighbor Particles
	/////

	// Count |ptc_num_neighbors|
	// - |radius|: searching radius
	__global__ void CountPtcNumNeighborsKernel(const float3* positions, 
		const int* cell_is_active_flags,
		const int* cell_to_active_cell_indices, const int* cell_ptc_indices, 
		const int* ptc_begins_in_active_cell, const int* active_cell_num_ptcs,
		const int num_ptcs, const float cell_sz, const int3 num_cells_dim,
		const float radius, int* ptc_num_neighbors)
	{
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		int3 ptc_cell = GetCell(positions[ptc_i], cell_sz);
		int num_neighbors = 0;
		const float radius_sqr = radius * radius;
		const float3 pos_i = positions[ptc_i];
		// We are only checking the 8 adjacent cells plus the cell itself,
		// this implies that our cell size must be greater than |radius|.
		for (int cz = -1; cz <= 1; ++cz) {
			for (int cy = -1; cy <= 1; ++cy) {
				for (int cx = -1; cx <= 1; ++cx) {
					int3 nb_cell = ptc_cell + make_int3(cx, cy, cz);
					if (!IsCellInRange(nb_cell, num_cells_dim))
						continue;
					int nb_cell_idx = GetCellIndex(nb_cell, num_cells_dim);
					if (!cell_is_active_flags[nb_cell_idx])
						continue;
					const int nb_ac_idx = 
						cell_to_active_cell_indices[nb_cell_idx];
					const int ac_num_ptcs = active_cell_num_ptcs[nb_ac_idx];
					const int nb_ptc_begin = ptc_begins_in_active_cell[nb_ac_idx];
					for (int offs = 0; offs < ac_num_ptcs; ++offs) {
						const int ptc_j = cell_ptc_indices[nb_ptc_begin + offs];
						if (ptc_i == ptc_j)
							continue;
						float dist_sqr = DistanceSquare(pos_i, positions[ptc_j]);
						if (dist_sqr < radius_sqr) {
							++num_neighbors;
						}
					}
				}
			}
		}
		ptc_num_neighbors[ptc_i] = num_neighbors;
	}
	
	// compute |ptc_neighbor_begins|
	void ComputePtcNeighborBegins(const d_vector<int>& ptc_num_neighbors,
		d_vector<int>* ptc_neighbor_begins) 
	{
		assert(ptc_num_neighbors.size() == 
			ptc_neighbor_begins->size());
		thrust::exclusive_scan(thrust::device, 
			ptc_num_neighbors.begin(), ptc_num_neighbors.end(),
			ptc_neighbor_begins->begin(), 0);
	}

	// Find neighbor particles and store them in |ptc_neighbor_indices|
	// - |radius|: searching radius
	__global__ void FindPtcNeighborIndicesKernel(const float3* positions,
		const int* cell_is_active_flags,
		const int* cell_to_active_cell_indices, const int* cell_ptc_indices, 
		const int* ptc_begins_in_active_cell, const int* active_cell_num_ptcs,
		const int num_ptcs, const float cell_sz, const int3 num_cells_dim,
		const float radius, const int* ptc_neighbor_begins, int* ptc_neighbor_indices, 
		const int* ptc_num_neighbors /*debug purpose, rm once correct*/)
	{
		const int ptc_i = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		int3 ptc_cell = GetCell(positions[ptc_i], cell_sz);
		int cur = ptc_neighbor_begins[ptc_i];
		const int cur_copy = cur;
		const float radius_sqr = radius * radius;
		const float3 pos_i = positions[ptc_i];
		// We are only checking the 8 adjacent cells plus the cell itself,
		// this implies that our cell size must be greater than |radius|.
		for (int cz = -1; cz <= 1; ++cz) {
			for (int cy = -1; cy <= 1; ++cy) {
				for (int cx = -1; cx <= 1; ++cx) {
					int3 nb_cell = ptc_cell + make_int3(cx, cy, cz);
					if (!IsCellInRange(nb_cell, num_cells_dim))
						continue;
					int nb_cell_idx = GetCellIndex(nb_cell, num_cells_dim);
					if (!cell_is_active_flags[nb_cell_idx])
						continue;
					const int nb_ac_idx = 
						cell_to_active_cell_indices[nb_cell_idx];
					const int ac_num_ptcs = active_cell_num_ptcs[nb_ac_idx];
					const int nb_ptc_begin = ptc_begins_in_active_cell[nb_ac_idx];
					for (int offs = 0; offs < ac_num_ptcs; ++offs) {
						const int ptc_j = cell_ptc_indices[nb_ptc_begin + offs];
						if (ptc_i == ptc_j)
							continue;
						float dist_sqr = DistanceSquare(pos_i, positions[ptc_j]);
						if (dist_sqr < radius_sqr) {
							ptc_neighbor_indices[cur] = ptc_j;
							++cur;
						}
					}
				}
			}
		}
		// Use GPU assert!
		// assert((cur - cur_copy) == ptc_num_neighbors[ptc_i]);
	}
	
	__global__ static void QueryCountKernel(const int num_cells, 
		const float3 range_min, const float3 range_max, const float3* positions, 
		const int* cell_is_active_flags, const int* cell_to_active_cell_indices,
		const int* ptc_begins_in_active_cell, const int* active_cell_num_ptcs,
		const int* cell_ptc_indices, int* cell_num_ptcs_inside) 
	{
		int cell_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (cell_i >= num_cells) return;
		
		bool is_active = cell_is_active_flags[cell_i];
		if (!is_active) return;

		const int ac_idx = cell_to_active_cell_indices[cell_i];
		const int ptc_begin = ptc_begins_in_active_cell[ac_idx];
		const int ac_num_ptcs = active_cell_num_ptcs[ac_idx];
		int num_inside = 0;
		for (int offs = 0; offs < ac_num_ptcs; ++offs) {
			int ptc_i = cell_ptc_indices[ptc_begin + offs];
			if (IsInside(positions[ptc_i], range_min, range_max)) {
				++num_inside;
			}
		}
		cell_num_ptcs_inside[cell_i] = num_inside;
	}

} // namespace impl_
		
	CellGridGpu::CellGridGpu(float3 world_sz, float cell_sz)
		: world_sz_per_dim_(world_sz), cell_sz_(cell_sz) {
		num_cells_per_dim_.x = (int)(world_sz_per_dim_.x / cell_sz_) + 1;
		num_cells_per_dim_.y = (int)(world_sz_per_dim_.y / cell_sz_) + 1;
		num_cells_per_dim_.z = (int)(world_sz_per_dim_.z / cell_sz_) + 1;

		total_num_cells_ = num_cells_per_dim_.x * num_cells_per_dim_.y
			* num_cells_per_dim_.z;
	}

	void UpdateCellGrid(const d_vector<float3>& positions, CellGridGpu* cell_grid)
	{
		using thrust::raw_pointer_cast;
		using namespace impl_;

		// extract necessary params
		const int num_ptcs = positions.size();
		const int num_cells = cell_grid->total_num_cells();

		// extract necessary pointers
		const float3* positions_ptr = raw_pointer_cast(positions.data());
		d_vector<int>& ptc_to_cell = cell_grid->ptc_to_cell;
		ptc_to_cell.clear();
		ptc_to_cell.resize(num_ptcs, 0);
		int* ptc_to_cell_ptr = raw_pointer_cast(ptc_to_cell.data());
		d_vector<int> cell_num_ptcs(num_cells, 0);
		int* cell_num_ptcs_ptr = raw_pointer_cast(cell_num_ptcs.data());
		// d_vector<int> ptc_offsets_within_cell(num_ptcs, 0);
		d_vector<int>& ptc_offsets_within_cell = 
			cell_grid->ptc_offsets_within_cell;
		ptc_offsets_within_cell.clear();
		ptc_offsets_within_cell.resize(num_ptcs, 0);
		int* ptc_offsets_within_cell_ptr = 
			raw_pointer_cast(ptc_offsets_within_cell.data());

		const int num_blocks_ptc = ComputeNumBlocks(num_ptcs);
		CellGridEntryPointKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
			positions_ptr, num_ptcs, cell_grid->cell_size(), 
			cell_grid->num_cells_per_dim(), ptc_to_cell_ptr, 
			cell_num_ptcs_ptr, ptc_offsets_within_cell_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		d_vector<int>& cell_is_active_flags = cell_grid->cell_is_active_flags;
		cell_is_active_flags.clear();
		cell_is_active_flags.resize(num_cells, 0);
		int* cell_is_active_flags_ptr = raw_pointer_cast(
			cell_is_active_flags.data());
		
		const int num_blocks_cell = ComputeNumBlocks(num_cells);
		SetCellIsActiveFlagsKernel<<<num_blocks_cell, kNumThreadPerBlock>>>(
			cell_num_ptcs_ptr, num_cells, cell_is_active_flags_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		d_vector<int>& cell_to_active_cell_indices = 
			cell_grid->cell_to_active_cell_indices;
		cell_to_active_cell_indices.clear();
		cell_to_active_cell_indices.resize(num_cells, 0);
		ComputeCellToActiveCellIndices(cell_is_active_flags,
			&cell_to_active_cell_indices);

		d_vector<int>& active_cell_num_ptcs = cell_grid->active_cell_num_ptcs;
		active_cell_num_ptcs.clear();
		active_cell_num_ptcs.resize(num_cells, 0);
		ComputeActiveCellNumPtcs(cell_num_ptcs, cell_is_active_flags,
			cell_to_active_cell_indices, &active_cell_num_ptcs);

		d_vector<int>& ptc_begins_in_active_cell =
			cell_grid->ptc_begins_in_active_cell;
		ptc_begins_in_active_cell.clear();
		ptc_begins_in_active_cell.resize(num_cells, 0);
		ComputePtcBeginsInActiveCell(
			active_cell_num_ptcs, &ptc_begins_in_active_cell);

		const int* cell_to_active_cell_indices_ptr =
			raw_pointer_cast(cell_to_active_cell_indices.data());
		const int* ptc_begins_in_active_cell_ptr =
			raw_pointer_cast(ptc_begins_in_active_cell.data());
		d_vector<int>& cell_ptc_indices = cell_grid->cell_ptc_indices;
		cell_ptc_indices.clear();
		cell_ptc_indices.resize(num_ptcs, 0);
		int* cell_ptc_indices_ptr = raw_pointer_cast(cell_ptc_indices.data());
		ComputeCellPtcIndicesKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
			ptc_to_cell_ptr, cell_to_active_cell_indices_ptr,
			ptc_begins_in_active_cell_ptr, ptc_offsets_within_cell_ptr,
			num_ptcs, cell_ptc_indices_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void FindParticleNeighbors(const d_vector<float3>& positions, const CellGridGpu& cell_grid, 
		const float h, GpuParticleNeighbors* pn) 
	{
		using namespace impl_;
		using thrust::raw_pointer_cast;
		// extract necessary params
		const float cell_sz = cell_grid.cell_size();
		const int3 num_cells_dim = cell_grid.num_cells_per_dim();
		const int num_ptcs = positions.size();
		// extract necessary pointers
		const float3* positions_ptr = raw_pointer_cast(positions.data());
		const int* cell_is_active_flags_ptr = raw_pointer_cast(cell_grid.cell_is_active_flags.data());
		const int* cell_to_active_cell_indices_ptr = 
			raw_pointer_cast(cell_grid.cell_to_active_cell_indices.data());
		const int* cell_ptc_indices_ptr = raw_pointer_cast(cell_grid.cell_ptc_indices.data());
		const int* ptc_begins_in_active_cell_ptr =
			raw_pointer_cast(cell_grid.ptc_begins_in_active_cell.data());
		const int* active_cell_num_ptcs_ptr = raw_pointer_cast(cell_grid.active_cell_num_ptcs.data());

		d_vector<int>& ptc_num_neighbors = pn->ptc_num_neighbors;
		// make sure we allocate memory first 
		ptc_num_neighbors.clear();
		ptc_num_neighbors.resize(num_ptcs, 0);
		int* ptc_num_neighbors_ptr = raw_pointer_cast(ptc_num_neighbors.data());
		
		const int num_blocks_ptc = ComputeNumBlocks(num_ptcs);
		// First step, count how many neighbors each particle has
		CountPtcNumNeighborsKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(positions_ptr,
			cell_is_active_flags_ptr,
			cell_to_active_cell_indices_ptr, cell_ptc_indices_ptr,
			ptc_begins_in_active_cell_ptr, active_cell_num_ptcs_ptr,
			num_ptcs, cell_sz, num_cells_dim,
			h, ptc_num_neighbors_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		
		// make sure we allocate memory first 
		d_vector<int>& ptc_neighbor_begins = pn->ptc_neighbor_begins;
		ptc_neighbor_begins.clear();
		ptc_neighbor_begins.resize(num_ptcs, 0);
		
		ComputePtcNeighborBegins(ptc_num_neighbors, &ptc_neighbor_begins);
		const int* ptc_neighbor_begins_ptr = raw_pointer_cast(ptc_neighbor_begins.data());

		// make sure we allocate memory first 
		d_vector<int>& ptc_neighbor_indices = pn->ptc_neighbor_indices;
		{
			int vec_sz = thrust::reduce(ptc_num_neighbors.begin(), ptc_num_neighbors.end(), 0);
			ptc_neighbor_indices.clear();
			ptc_neighbor_indices.resize(vec_sz, 0);
		}
		int* ptc_neighbor_indices_ptr = raw_pointer_cast(ptc_neighbor_indices.data());

		// Second step, record the neighbor indices for each particle
		FindPtcNeighborIndicesKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(positions_ptr,
			cell_is_active_flags_ptr,
			cell_to_active_cell_indices_ptr, cell_ptc_indices_ptr, ptc_begins_in_active_cell_ptr, 
			active_cell_num_ptcs_ptr, num_ptcs, cell_sz, num_cells_dim, h, ptc_neighbor_begins_ptr, 
			ptc_neighbor_indices_ptr, ptc_num_neighbors_ptr /*debug purpose, rm once correct*/);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void Query(const d_vector<float3>& positions, const CellGridGpu& cell_grid, 
		const AABB& range, d_vector<int>* cell_num_ptcs_inside) {
		using namespace impl_;
		using thrust::raw_pointer_cast;

		const int num_cells = cell_grid.total_num_cells();
		const float3 range_min = Convert(range.min());
		const float3 range_max = Convert(range.max());
		const float3* positions_ptr = raw_pointer_cast(positions.data());
		const int* cell_is_active_flags_ptr =
			raw_pointer_cast(cell_grid.cell_is_active_flags.data());
		const int* cell_to_active_cell_indices_ptr =
			raw_pointer_cast(cell_grid.cell_to_active_cell_indices.data());
		const int* ptc_begins_in_active_cell_ptr =
			raw_pointer_cast(cell_grid.ptc_begins_in_active_cell.data());
		const int* active_cell_num_ptcs_ptr =
			raw_pointer_cast(cell_grid.active_cell_num_ptcs.data());
		const int* cell_ptc_indices_ptr = 
			raw_pointer_cast(cell_grid.cell_ptc_indices.data());
		cell_num_ptcs_inside->clear();
		cell_num_ptcs_inside->resize(num_cells, 0);
		int* cell_num_ptcs_inside_ptr =
			raw_pointer_cast(cell_num_ptcs_inside->data());

		const int num_blocks_cell = ComputeNumBlocks(num_cells);
		QueryCountKernel<<<num_blocks_cell, kNumThreadPerBlock>>>(
			num_cells, range_min, range_max, positions_ptr, 
			cell_is_active_flags_ptr, cell_to_active_cell_indices_ptr,
			ptc_begins_in_active_cell_ptr, active_cell_num_ptcs_ptr,
			cell_ptc_indices_ptr, cell_num_ptcs_inside_ptr);
	}

#define GRAVITY_Y -9.8f
	__global__ static void ApplyGravityKernel(const int num_ptcs, const float dt,
		float3* positions, float3* velocities) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		float3 pos_i = positions[ptc_i];
		float3 vel_i = velocities[ptc_i];

		vel_i.y += GRAVITY_Y * dt;
		pos_i += vel_i * dt;
		
		positions[ptc_i] = pos_i;
		velocities[ptc_i] = vel_i;
	}
#undef GRAVITY_Y

	__global__ static void ImposeBoundaryConstraintKernel(const int num_ptcs, const float world_size,
		const float board_x, const float board_vel_x,
		float3* positions, float3* velocities)
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		float3 pos_i = positions[ptc_i];
		float3 vel_i = velocities[ptc_i];
#define IMPOSE_ON_DIM(D) \
	if (pos_i.##D <= 0.0f || pos_i.##D >= world_size) { \
		vel_i.##D = 0.0f; \
		pos_i.##D = max(0.0f, min(world_size, pos_i.##D)); \
	}
		// IMPOSE_ON_DIM(x);
		IMPOSE_ON_DIM(y);
		IMPOSE_ON_DIM(z);
#undef IMPOSE_ON_DIM
		if (pos_i.x <= 0.0f) {
			vel_i.x = 0.0f;
			pos_i.x = 0.0f;
		}
		else if (pos_i.x >= board_x) {
			float vel_x_rel = vel_i.x - board_vel_x;
			if (vel_x_rel > 0.0f) {
				vel_i.x = board_vel_x;
			}
			pos_i.x = board_x;
		}
		positions[ptc_i] = pos_i;
		velocities[ptc_i] = vel_i;
	}

	__global__ static void ComputeLambdaKernel(const float3* positions, const int* ptc_num_neighbors,
		const int* ptc_neighbor_begins, const int* ptc_neighbor_indices, const int num_ptcs,
		const float h, const float mass, const float rho_0_recpr, const float epsilon, float* lambdas) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		const int num_nbs = ptc_num_neighbors[ptc_i];
		const int nb_begin = ptc_neighbor_begins[ptc_i];
		const float3 pos_i = positions[ptc_i];

		float3 gradient_i = make_float3(0.0f);
		float sum_gradient = 0.0f;
		float density_constraint = 0.0f;
		for (int offs = 0; offs < num_nbs; ++offs) {
			const int ptc_j = ptc_neighbor_indices[nb_begin + offs];
			const float3 pos_ji = pos_i - positions[ptc_j];
			
			const float3 gradient_j = SpikyGradient(pos_ji, h);
			sum_gradient += dot(gradient_j, gradient_j);
			gradient_i += gradient_j;
			
			density_constraint += mass * Poly6Value(pos_ji, h);
		}
		sum_gradient += dot(gradient_i, gradient_i);
		density_constraint = (density_constraint * rho_0_recpr) - 1.0f;

		const float lambda_i = (-density_constraint) / (sum_gradient + epsilon);
		lambdas[ptc_i] = lambda_i;
	}
	
	__device__ float ComputeScorr(const float3 pos_ji, const float h, 
		const float corr_delta_q_coeff, const float corr_k, const float corr_n) {
		// Eq (13)
		float x = Poly6Value(pos_ji, h) / Poly6Value(corr_delta_q_coeff * h, h);
		float result = (-corr_k) * pow(x, corr_n);
		return result;
	}
	
	__global__ static void ComputeDeltaPositionsKernel(const float3* positions,
		const int* ptc_num_neighbors, const int* ptc_neighbor_begins, const int* ptc_neighbor_indices, 
		const float* lambdas, const int num_ptcs, const float h, const float rho_0_recpr,
		const float corr_delta_q_coeff, const float corr_k, const float corr_n, float3* delta_positions) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		const int num_nbs = ptc_num_neighbors[ptc_i];
		const int nb_begin = ptc_neighbor_begins[ptc_i];
		const float3 pos_i = positions[ptc_i];
		const float lambda_i = lambdas[ptc_i];

		float3 delta_pos_i = make_float3(0.0f);
		for (int offs = 0; offs < num_nbs; ++offs) {
			const int ptc_j = ptc_neighbor_indices[nb_begin + offs];
			const float lambda_j = lambdas[ptc_j];
			const float3 pos_ji = pos_i - positions[ptc_j];
			const float scorr_ij = ComputeScorr(pos_ji, h, corr_delta_q_coeff, corr_k, corr_n);
			delta_pos_i += (lambda_i + lambda_j + scorr_ij) * SpikyGradient(pos_ji, h);
		}
		delta_pos_i *= rho_0_recpr;
		delta_positions[ptc_i] = delta_pos_i;
	}
	
	__global__ static void ApplyDeltaPositionsKernel(const float3* delta_positions,
		const int num_ptcs, float3* positions)
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		float3 pos_i = positions[ptc_i];
		pos_i += delta_positions[ptc_i];
		positions[ptc_i] = pos_i;
	}

	__global__ static void UpdateVelocitiesKernel(const float3* old_positions, const float3* new_positions,
		const int num_ptcs, const float dt, float3* velocities) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		const float3 old_pos_i = old_positions[ptc_i];
		const float3 new_pos_i = new_positions[ptc_i];
		const float3 new_vel_i = (new_pos_i - old_pos_i) / dt;
		velocities[ptc_i] = new_vel_i;
	}

	__global__ static void ComputeVorticitiesKernel(const float3* positions, const float3* velocities,
		const int* ptc_num_neighbors, const int* ptc_neighbor_begins, const int* ptc_neighbor_indices, 
		const int num_ptcs, const float h, float3* vorticities) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		const float3 pos_i = positions[ptc_i];
		const float3 vel_i = velocities[ptc_i];
		const int num_nbs = ptc_num_neighbors[ptc_i];
		const int nb_begin = ptc_neighbor_begins[ptc_i];
		
		float3 vorticity = make_float3(0.0f);
		for (int offs = 0; offs < num_nbs; ++offs) {
			const int ptc_j = ptc_neighbor_indices[nb_begin + offs];
			// vel_diff_ij = vel_j - vel_i;
			const float3 vel_ij = velocities[ptc_j] - vel_i;
			// gradient = kernel_.Gradient(pos_i - pos_j);
			const float3 pos_ji = pos_i - positions[ptc_j];
			const float3 gradient = SpikyGradient(pos_ji, h);
			// result += glm::cross(vel_diff_ij, gradient);
			vorticity += cross(vel_ij, gradient);
		}
		vorticities[ptc_i] = vorticity;
	}

	__global__ static void ComputeVorticityCorrForcesKernel(const float3* positions, 
		const int* ptc_num_neighbors, const int* ptc_neighbor_begins, const int* ptc_neighbor_indices,
		const float3* vorticities, const int num_ptcs, const float h, const float vorticity_epsilon,
		float3* vorticity_corr_forces) 
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;
		
		const float3 pos_i = positions[ptc_i];
		// Compute Eta
		const int num_nbs = ptc_num_neighbors[ptc_i];
		const int nb_begin = ptc_neighbor_begins[ptc_i];
		float3 eta = make_float3(0.0f);
		for (int offs = 0; offs < num_nbs; ++offs) {
			const int ptc_j = ptc_neighbor_indices[nb_begin + offs];
			const float3 pos_ji = pos_i - positions[ptc_j];
			const float omega_j_len = length(vorticities[ptc_j]);
			const float3 gradient = SpikyGradient(pos_ji, h);
			eta += (omega_j_len * gradient);
		}
		// Compute Vorticity Corr Force
		const float eta_len = length(eta);
		float3 vort_corr_force = make_float3(0.0f);
		if (eta_len > 1e-6) {
			eta = normalize(eta);
			const float3 omega_i = vorticities[ptc_i];
			vort_corr_force = vorticity_epsilon * cross(eta, omega_i);
		}
		vorticity_corr_forces[ptc_i] = vort_corr_force;
	}

	__global__ static void ComputeXsphsKernel(const float3* positions, const float3* velocities,
		const int* ptc_num_neighbors, const int* ptc_neighbor_begins, const int* ptc_neighbor_indices,
		const int num_ptcs, const float h, const float xsph_c, float3* xsphs)
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		const float3 pos_i = positions[ptc_i];
		const float3 vel_i = velocities[ptc_i];
		const int num_nbs = ptc_num_neighbors[ptc_i];
		const int nb_begin = ptc_neighbor_begins[ptc_i];

		float3 xsph = make_float3(0.0f);
		for (int offs = 0; offs < num_nbs; ++offs) {
			const int ptc_j = ptc_neighbor_indices[nb_begin + offs];
			const float3 vel_ij = velocities[ptc_j] - vel_i;
			const float w = Poly6Value(pos_i - positions[ptc_j], h);
			xsph += (w * vel_ij);
		}
		xsph *= xsph_c;
		xsphs[ptc_i] = xsph;
	}

	__global__ static void ApplyVelocityCorrectionsKernel(const float3* vorticity_corr_forces, 
		const float3* xsphs, const int num_ptcs, const float dt, float3* velocities)
	{
		const int ptc_i = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (ptc_i >= num_ptcs) return;

		float3 vel_i = velocities[ptc_i];
		vel_i += vorticity_corr_forces[ptc_i] * dt;
		vel_i += xsphs[ptc_i];
		velocities[ptc_i] = vel_i;
	}
	
	void PbfSolverGpu::CustomConfigure_(const PbfSolverConfig& config) {
		cell_grid_size_ = config.spatial_hash_cell_size;

		board_x_ = world_size_ - 1.0f;
		board_x_vel_ = -5.0f;
	}
	
	void PbfSolverGpu::CustomInitPs_() {
		// cache num of particles
		num_ptcs_ = ps_->NumParticles();

		// copy the positions/velocities to the device memory.
		//
		// This is only needed once, later on after every update,
		// we only need to copy the most up-to-date data from the
		// device memory back to each particle. This is because
		// PbfSolverGpu is the only one who modifies the particles'
		// data (position/velocity).
		d_positions_.reserve(num_ptcs_);
		d_velocities_.reserve(num_ptcs_);
		for (size_t p_i = 0; p_i < num_ptcs_; ++p_i) {
			auto ptc_i = ps_->Get(p_i);
			d_positions_.push_back(Convert(ptc_i.position()));
			d_velocities_.push_back(Convert(ptc_i.velocity()));
		}
		
		// init particle records
		const float3 zeros = make_float3(0.0f);
		old_positions_.resize(num_ptcs_, zeros);
		lambdas_.resize(num_ptcs_, 0.0f);
		delta_positions_.resize(num_ptcs_, zeros);
		vorticities_.resize(num_ptcs_, zeros);
		vorticity_corr_forces_.resize(num_ptcs_, zeros);
		xsphs_.resize(num_ptcs_, zeros);
	}
	
	void PbfSolverGpu::Update(float dt) {

		board_x_ += board_x_vel_ * dt;
		bool change_board_x_vel_dir = false;
		if (board_x_ < world_size_ * 0.5f) {
			board_x_ = world_size_ * 0.5f;
			change_board_x_vel_dir = true;
		}
		else if (board_x_ > world_size_ - 0.5f) {
			board_x_ = world_size_ - 0.5f;
			change_board_x_vel_dir = true;
		}
		if (change_board_x_vel_dir)
			board_x_vel_ = -board_x_vel_;

		ResetParticleRecords_();
		RecordOldPositions_();

		ApplyGravity_(dt);
		ImposeBoundaryConstraint_();
		
		FindNeighbors_();
		
		for (unsigned itr = 0; itr < num_iters_; ++itr) {
			ComputeLambdas_();
			ComputeDeltaPositions_();
			ApplyDeltaPositions_();
		}
		
		ImposeBoundaryConstraint_();
		UpdateVelocities_(dt);

		ComputeVorticities_();
		ComputeVorticityCorrForces_();
		ComputeXsphs_();
		ApplyVelocityCorrections_(dt);

		UpdatePs_();
	}
	
	void PbfSolverGpu::ResetParticleRecords_() {
		using thrust::fill;
		using thrust::device;

		const float3 zeros = make_float3(0.0f);
		fill(device, old_positions_.begin(), old_positions_.end(), zeros);
		fill(device, lambdas_.begin(), lambdas_.end(), 0.0f);
		fill(device, delta_positions_.begin(), delta_positions_.end(), zeros);
		fill(device, vorticities_.begin(), vorticities_.end(), zeros);
		fill(device, vorticity_corr_forces_.begin(), vorticity_corr_forces_.end(), zeros);
		fill(device, xsphs_.begin(), xsphs_.end(), zeros);
	}
	
	void PbfSolverGpu::RecordOldPositions_() {
		thrust::copy(thrust::device, d_positions_.begin(), d_positions_.end(), old_positions_.begin());
	}
		
	void PbfSolverGpu::ApplyGravity_(const float dt) {
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ApplyGravityKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(num_ptcs_, dt,
			PositionsPtr_(), VelocitiesPtr_());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void PbfSolverGpu::ImposeBoundaryConstraint_() {
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ImposeBoundaryConstraintKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
			num_ptcs_, world_size_, board_x_, board_x_vel_, PositionsPtr_(), VelocitiesPtr_());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::FindNeighbors_() {
		const float3 world_sz_dim = make_float3(world_size_);
		
		CellGridGpu cell_grid{ world_sz_dim, cell_grid_size_ };
		UpdateCellGrid(d_positions_, &cell_grid);
			
		FindParticleNeighbors(d_positions_, cell_grid, h_, &ptc_nb_recs_);
	}

	void PbfSolverGpu::ComputeLambdas_() {
		float* lambdas_ptr = thrust::raw_pointer_cast(lambdas_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ComputeLambdaKernel<<<num_blocks_ptc, kNumThreadPerBlock>>> (
			PositionsPtr_(), ptc_nb_recs_.ptc_num_neighbors_ptr(), ptc_nb_recs_.ptc_neighbor_begins_ptr(), 
			ptc_nb_recs_.ptc_neighbor_indices_ptr(), num_ptcs_, h_, mass_, rho_0_recpr_, epsilon_, lambdas_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::ComputeDeltaPositions_() {
		const float* lambdas_ptr = thrust::raw_pointer_cast(lambdas_.data());
		float3* delta_positions_ptr = thrust::raw_pointer_cast(delta_positions_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ComputeDeltaPositionsKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
			PositionsPtr_(), ptc_nb_recs_.ptc_num_neighbors_ptr(), ptc_nb_recs_.ptc_neighbor_begins_ptr(),
			ptc_nb_recs_.ptc_neighbor_indices_ptr(),  lambdas_ptr, num_ptcs_, h_, rho_0_recpr_, 
			corr_delta_q_coeff_, corr_k_, corr_n_, delta_positions_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::ApplyDeltaPositions_() {
		const float3* delta_positions_ptr = thrust::raw_pointer_cast(delta_positions_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ApplyDeltaPositionsKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(
			delta_positions_ptr, num_ptcs_, PositionsPtr_());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::UpdateVelocities_(const float dt) {
		const float3* old_positions_ptr = thrust::raw_pointer_cast(old_positions_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		UpdateVelocitiesKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(old_positions_ptr, PositionsPtr_(),
			num_ptcs_, dt, VelocitiesPtr_());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void PbfSolverGpu::ComputeVorticities_() {
		float3* vorticities_ptr = thrust::raw_pointer_cast(vorticities_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ComputeVorticitiesKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(PositionsPtr_(), VelocitiesPtr_(),
			ptc_nb_recs_.ptc_num_neighbors_ptr(), ptc_nb_recs_.ptc_neighbor_begins_ptr(),
			ptc_nb_recs_.ptc_neighbor_indices_ptr(), num_ptcs_, h_, vorticities_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::ComputeVorticityCorrForces_() {
		const float3* vorticities_ptr = thrust::raw_pointer_cast(vorticities_.data());
		float3* vort_corr_forces_ptr = thrust::raw_pointer_cast(vorticity_corr_forces_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);

		ComputeVorticityCorrForcesKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(PositionsPtr_(),
			ptc_nb_recs_.ptc_num_neighbors_ptr(), ptc_nb_recs_.ptc_neighbor_begins_ptr(),
			ptc_nb_recs_.ptc_neighbor_indices_ptr(), vorticities_ptr, num_ptcs_, h_, vorticity_epsilon_,
			vort_corr_forces_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	void PbfSolverGpu::ComputeXsphs_() {
		float3* xsphs_ptr = thrust::raw_pointer_cast(xsphs_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ComputeXsphsKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(PositionsPtr_(), VelocitiesPtr_(),
			ptc_nb_recs_.ptc_num_neighbors_ptr(), ptc_nb_recs_.ptc_neighbor_begins_ptr(),
			ptc_nb_recs_.ptc_neighbor_indices_ptr(), num_ptcs_, h_, xsph_c_, xsphs_ptr);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::ApplyVelocityCorrections_(const float dt) {
		const float3* vort_corr_forces_ptr = thrust::raw_pointer_cast(vorticity_corr_forces_.data());
		const float3* xsphs_ptr = thrust::raw_pointer_cast(xsphs_.data());
		const int num_blocks_ptc = impl_::ComputeNumBlocks(num_ptcs_);
		ApplyVelocityCorrectionsKernel<<<num_blocks_ptc, kNumThreadPerBlock>>>(vort_corr_forces_ptr,
			xsphs_ptr, num_ptcs_, dt, VelocitiesPtr_());
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
	
	void PbfSolverGpu::UpdatePs_() {
		using thrust::host_vector;
		host_vector<float3> h_positions{ d_positions_ };
		host_vector<float3> h_velocities{ d_velocities_ };

		for (size_t p_i = 0; p_i < num_ptcs_; ++p_i) {
			auto ptc_i = ps_->Get(p_i);
			ptc_i.set_position(Convert(h_positions[p_i]));
			ptc_i.set_velocity(Convert(h_velocities[p_i]));
		}

	}
} // namespace pbf