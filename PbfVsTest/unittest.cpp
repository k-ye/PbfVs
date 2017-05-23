#include "CppUnitTest.h"
#include "stdafx.h"

#include "include/aabb.h"
#include "include/particle_system.h"
#include "include/sh_position_getter.h"
#include "include/spatial_hash.h"

#include <sstream>
#include <stdlib.h> // srand, rand
#include <string>
#include <time.h> // time
#include <unordered_set>
#include <vector>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

// How to properly build the application project in the unit test project.
// http://stackoverflow.com/questions/19886397/how-to-solve-the-error-lnk2019-unresolved-external-symbol-function

namespace pbf 
{		
	TEST_CLASS(SpatialHashTest)
	{
	public:
		SpatialHashTest() 
			: kNumPoints(1000u), kCellSize(1.0f), kHalfCellSize(kCellSize / 2), 
			kNumCellsPerDim(12u), kAabbOffsetByCell(3u), kNumIters(100u) {
			const point_t kAabbMin{ kCellSize * kAabbOffsetByCell };
			const point_t kAabbMax{ kCellSize * (kNumCellsPerDim - kAabbOffsetByCell) };
			query_aabb_ = AABB{ kAabbMin, kAabbMax };
		}

		TEST_METHOD(TestCorrectness)
		{
			srand(time(nullptr));
			Init();

			for (int iter = 0; iter < kNumIters; ++iter) {
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
		}

	private:
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

		const unsigned kNumPoints;
		const float kCellSize;
		const float kHalfCellSize;
		const unsigned kNumCellsPerDim;
		const unsigned kAabbOffsetByCell;
		const unsigned kNumIters;

		AABB query_aabb_;
		ParticleSystem ps_;
		SpatialHash<size_t, PositionGetter> spatial_hash_;
		std::unordered_set<size_t> ptcs_inside_aabb_ref_;
	};
}