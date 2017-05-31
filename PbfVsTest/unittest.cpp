#include "CppUnitTest.h"
#include "stdafx.h"

#include "include\aabb.h"
#include "include\particle_system.h"
#include "include\sh_position_getter.h"
#include "include\spatial_hash.h"
#include "include\cuda_basic.h"

#include "include\pbf_solver_gpu.h"

#include <thrust\execution_policy.h>
#include <thrust\host_vector.h>
#include <sstream>
#include <stdlib.h> // srand, rand
#include <string>
#include <time.h> // time
#include <unordered_set>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

// How to properly build the application project in the unit test project.
// http://stackoverflow.com/questions/19886397/how-to-solve-the-error-lnk2019-unresolved-external-symbol-function

namespace pbf {
namespace {
	// The testing world is a cube of equal size in three dimensions.
	// Each cube is consisted of a series of cells. These cells are
	// not the same thing as the cell in the data structure we are
	// testing on.
	
	const unsigned kNumPoints = 1000u;
	// Cell size of the test world.
	const float kCellSize = 1.0f;
	const float kHalfCellSize = kCellSize / 2;
	const unsigned kNumCellsPerDim = 15u;
	const float kWorldSize = kCellSize * kNumCellsPerDim;
	const unsigned kAabbOffsetByCell = 3u;
	const unsigned kNumIters = 100u;
	// Cell size of the data structure being tested.
	const float kTestDsCellSize = 1.5f;

	float GenRandomFloat(float lo, float hi) {
		float result = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (hi - lo)) + lo;
		return result;
	}

	point_t GenRandomPoint() {
		int x = rand() % kNumCellsPerDim;
		int y = rand() % kNumCellsPerDim;
		int z = rand() % kNumCellsPerDim;
		
		point_t result;
		result.x = x * kCellSize + kHalfCellSize;
		result.y = y * kCellSize + kHalfCellSize;
		result.z = z * kCellSize + kHalfCellSize;
		return result;
	}

	AABB GetQueryAABB() {
		point_t kAabbMin{ kCellSize * kAabbOffsetByCell };
		point_t kAabbMax{ kCellSize * (kNumCellsPerDim - 
			kAabbOffsetByCell) };
		AABB aabb{ kAabbMin, kAabbMax };
		return aabb;
	}

	int Reduce(const d_vector<int>& d_vec) {
		thrust::host_vector<int> h_vec{ d_vec };
		int result = 0;
		for (int i : h_vec)
			result += i;
		return result;
	}
} // namespace anonymous
	
	TEST_CLASS(SpatialHashTest)
	{
	public:
		SpatialHashTest() : query_aabb_(GetQueryAABB()) { }

		TEST_METHOD(TestSpatialHashCorrect)
		{
			srand(time(nullptr));
			Init();

			for (int iter = 0; iter < kNumIters; ++iter) {
				TestShOneIter();
			}
		}

	private:
		void TestShOneIter() {
			RandomScatterPoints();
			spatial_hash_.UpdateAll();
			auto query_result = spatial_hash_.Query(query_aabb_);

			std::stringstream ss;
			ss << "Query result size: " << query_result.size();
			auto log_str = ss.str();
			Logger::WriteMessage(log_str.c_str());
			// Assert::AreEqual(query_result.size(), num_inside_aabb_ref_);
			Assert::AreEqual(query_result.size(), ptcs_inside_aabb_ref_.size());
			for (size_t ptc_i : query_result) {
				Assert::IsTrue(ptcs_inside_aabb_ref_.count(ptc_i) == 1);
			}
		}
		
		void Init() {
			// init particle system
			for (unsigned i = 0; i < kNumPoints; ++i) {
				ps_.Add(point_t{ 0.0f }, point_t{ 0.0f });
			}

			// init spatial hash
			spatial_hash_.set_cell_size(0.583f);
			PositionGetter pg{ &ps_ };
			spatial_hash_.set_pos_getter(pg);
			spatial_hash_.Clear();
			for (size_t i = 0; i < ps_.NumParticles(); ++i) {
				spatial_hash_.Add(i);
			}
		}

		void RandomScatterPoints() {
			Assert::AreEqual(kNumPoints, ps_.NumParticles());
			
			ptcs_inside_aabb_ref_.clear();
			for (unsigned ptc_i = 0; ptc_i < kNumPoints; ++ptc_i) {
				auto ptc = ps_.Get(ptc_i);
				auto pos = GenRandomPoint();
				ptc.set_position(pos);
				if (query_aabb_.Contains(pos)) {
					ptcs_inside_aabb_ref_.insert(ptc_i);
				}
			}
		}

		AABB query_aabb_;
		ParticleSystem ps_;
		SpatialHash<size_t, PositionGetter> spatial_hash_;
		std::unordered_set<size_t> ptcs_inside_aabb_ref_;
	};
	
	TEST_CLASS(CellGridGpuTest)
	{
	public:
		CellGridGpuTest() : query_aabb_(GetQueryAABB()) {}

		TEST_METHOD(TestCellGridGpu)
		{
			srand(time(nullptr));

			for (int iter = 0; iter < kNumIters; ++iter) {
				TestCellGridGpuOneIter();
			}
		}

		TEST_METHOD(TestFindNeighbors)
		{
			std::vector<float3> h_positions;
			h_positions.push_back(make_float3(0, 0, 0));
			auto gen_rand_float3 = [](float lo, float hi) -> float3 {
				float x = GenRandomFloat(lo, hi);
				float y = GenRandomFloat(lo, hi);
				float z = GenRandomFloat(lo, hi);
				return make_float3(x, y, z);
			};
			for (int i = 0; i < 10; ++i) {
				h_positions.push_back(gen_rand_float3(0, 0.5f));
			}
			for (int i = 0; i < 10; ++i) {
				h_positions.push_back(gen_rand_float3(2.0f, 4.0f));
			}
			
			d_vector<float3> d_positions{ h_positions };
			float3 world_sz_dim = make_float3(5.0f, 5.0f, 5.0f);
			CellGridGpu cell_grid{ world_sz_dim, 1.5f /* kTestDsCellSize */};
			const float cell_sz = cell_grid.cell_size();
			const int3 num_cells_dim = cell_grid.num_cells_per_dim();
			const float h = 1.0f;

			UpdateCellGrid(d_positions, &cell_grid);
			
			ParticleNeighbors pn;
			FindParticleNeighbors(d_positions, cell_grid, cell_sz, num_cells_dim, h, &pn);
			thrust::host_vector<int> h_ptc_num_neighbors{ pn.ptc_num_neighbors };
			std::stringstream ss;
			ss << "Num neighbors ref size: " << 10 
				<< ", cuda computed size: " << h_ptc_num_neighbors[0];
			auto log_str = ss.str();
			Logger::WriteMessage(log_str.c_str());
		}

	private:
		void TestCellGridGpuOneIter() {
			using thrust::host_vector;
			
			std::vector<float3> h_positions;
			RandomScatterPoints(&h_positions);

			d_vector<float3> d_positions{ h_positions };
			float3 world_sz_dim = make_float3(kWorldSize, kWorldSize, kWorldSize);
			CellGridGpu cell_grid{ world_sz_dim, kTestDsCellSize };

			UpdateCellGrid(d_positions, &cell_grid);

#if 0
			// More verbosed test
			host_vector<int> h_ptc_to_cell{ cell_grid.ptc_to_cell };
			host_vector<int> h_cell_to_active_cell_indices{
				cell_grid.cell_to_active_cell_indices };
			host_vector<int> h_ptc_begins_in_active_cell{
				cell_grid.ptc_begins_in_active_cell };
			host_vector<int> h_ptc_offsets_within_cell{ 
				cell_grid.ptc_offsets_within_cell };
			host_vector<int> h_cell_ptc_indices{ cell_grid.cell_ptc_indices };
			for (int ptc_i = 0; ptc_i < kNumPoints; ++ptc_i) {
				int cell_i = h_ptc_to_cell[ptc_i];
				int ac_idx = h_cell_to_active_cell_indices[cell_i];
				int ac_ptc_begin = h_ptc_begins_in_active_cell[ac_idx];
				int offs = h_ptc_offsets_within_cell[ptc_i];
				int ptc_i_prime = h_cell_ptc_indices[ac_ptc_begin + offs];
				Assert::AreEqual(ptc_i, ptc_i_prime);
			}
#endif
			d_vector<int> cell_num_ptcs_inside;
			Query(d_positions, cell_grid, query_aabb_, &cell_num_ptcs_inside);
			int num_ptcs_inside = Reduce(cell_num_ptcs_inside);
			std::stringstream ss;
			ss << "Query ref size: " << ptcs_inside_aabb_ref_.size()
				<< ", cuda computed size: " << num_ptcs_inside;
			auto log_str = ss.str();
			Logger::WriteMessage(log_str.c_str());
			ss.str("");
			Assert::AreEqual(num_ptcs_inside, (int)ptcs_inside_aabb_ref_.size());

		}
		
		void RandomScatterPoints(std::vector<float3>* positions) {
			positions->clear();
			ptcs_inside_aabb_ref_.clear();

			for (unsigned ptc_i = 0; ptc_i < kNumPoints; ++ptc_i) {
				auto pos = GenRandomPoint();
				positions->push_back(Convert(pos));
				if (query_aabb_.Contains(pos)) {
					ptcs_inside_aabb_ref_.insert(ptc_i);
				}
			}
		}

		AABB query_aabb_;
		std::unordered_set<size_t> ptcs_inside_aabb_ref_;
	};
}