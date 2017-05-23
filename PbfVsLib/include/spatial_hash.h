//
//  spatial_hash.h
//  PBF
//
//  Created by Ye Kuang on 3/31/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef spatial_hash_h
#define spatial_hash_h

#include "basic.h"

#include "aabb.h"

#include <algorithm>        // std::swap
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pbf
{
	template <typename T>
	struct hash
	{
	public:
		typedef glm::tvec3<T> vec3_t;

		size_t operator()(const vec3_t& v) const
		{
			size_t result = 0;
			result = Hash_(v[0], result);
			result = Hash_(v[1], result);
			result = Hash_(v[2], result);
			return result;
		}

	private:
		inline size_t Hash_(T val, size_t seed) const
		{
			size_t result = seed;
			result ^= std::hash<T>{}(val)+0x9e3779b9 + (seed << 6) + (seed >> 2);
			return result;
		}
	};

	template <typename T, typename PG>
	class SpatialHash
	{
	public:
		typedef T data_t;
		typedef unsigned cell_comp_t;
		typedef glm::tvec3<cell_comp_t> cell_t;
		typedef std::vector<size_t> query_t;

	public:
		void set_cell_size(float s)
		{
			cell_size_ = s;
			cell_size_recpr_ = 1.0f / s;
		}
		void set_pos_getter(const PG& pg) { pos_getter_ = pg; }

		size_t size() const { return records_.size(); }

		size_t Add(const data_t& dat)
		{
			cell_t cell = ComputeCell_(pos_getter_(dat));
			size_t cell_hash = cell_hasher_(cell);

			size_t rec_index = records_.size();
			size_t cell_hash_index = hash2idxs_[cell_hash].size();

			Record rec;
			rec.user_data = dat;
			rec.cell_hash = cell_hash;
			rec.cell_hash_index = cell_hash_index;

			records_.push_back(rec);
			hash2idxs_[cell_hash].push_back(rec_index);

			return rec_index;
		}

		void Clear()
		{
			records_.clear();
			hash2idxs_.clear();
		}

		// \invariant: After Update(), the record is still stored
		// at index @i.
		void Update(size_t i, const data_t& dat)
		{
			/// remove
			Record& rec = records_[i];
			size_t old_cell_hash = rec.cell_hash;
			size_t old_cell_hash_index = rec.cell_hash_index;

			auto& cell_idxs = hash2idxs_[old_cell_hash];
			assert(cell_idxs.size() > 0);

			// Swap integers, just use the std version.
			size_t back_rec_idx = cell_idxs.back();
			records_[back_rec_idx].cell_hash_index = old_cell_hash_index;
			cell_idxs[old_cell_hash_index] = back_rec_idx;
			// std::swap(cell_idxs[old_cell_hash_index], cell_idxs.back());
			cell_idxs.pop_back();

			/// then update
			cell_t new_cell = ComputeCell_(pos_getter_(dat));
			size_t new_cell_hash = cell_hasher_(new_cell);
			size_t new_cell_hash_index = hash2idxs_[new_cell_hash].size();

			rec.user_data = dat;
			rec.cell_hash = new_cell_hash;
			rec.cell_hash_index = new_cell_hash_index;

			hash2idxs_[new_cell_hash].push_back(i);
		}

		// \invariant: After Update(), the record is still stored
		// at index @i.
		void Update(size_t i)
		{
			const Record& rec = records_[i];
			Update(i, rec.user_data);
		}

		void UpdateAll()
		{
			for (size_t i = 0; i < size(); ++i)
				Update(i);
		}

		data_t Get(size_t i) const { return records_[i].user_data; }
		const data_t& GetRef(size_t i) const { return records_[i].user_data; }

		query_t Query(const AABB& range) const
		{
			std::unordered_set<size_t> tmp_result;

			cell_t min_cell = ComputeCell_(range.min());
			cell_t max_cell = ComputeCell_(range.max());

			for (cell_comp_t it_x = min_cell.x; it_x <= max_cell.x; ++it_x)
			{
				for (cell_comp_t it_y = min_cell.y; it_y <= max_cell.y; ++it_y)
				{
					for (cell_comp_t it_z = min_cell.z; it_z <= max_cell.z; ++it_z)
					{
						cell_t it_cell{ it_x, it_y, it_z };
						size_t it_cell_hash = cell_hasher_(it_cell);

						auto hash_idxs_it = hash2idxs_.find(it_cell_hash);
						if (hash_idxs_it != hash2idxs_.end())
						{
							for (size_t rec_index : hash_idxs_it->second)
							{
								auto rec_pos = pos_getter_(records_[rec_index].user_data);
								if (range.Contains(rec_pos))
									tmp_result.insert(rec_index);
							}
						}
					}
				}
			}

			query_t result{ tmp_result.begin(), tmp_result.end() };
			return result;
		}

	private:
		cell_t ComputeCell_(const point_t& pos) const
		{
			cell_t result;
			result[0] = (cell_comp_t)pos[0] * cell_size_recpr_;
			result[1] = (cell_comp_t)pos[1] * cell_size_recpr_;
			result[2] = (cell_comp_t)pos[2] * cell_size_recpr_;
			return result;
		};

		struct Record
		{
			data_t user_data;
			size_t cell_hash;
			size_t cell_hash_index;
		};

	private:
		float cell_size_;
		float cell_size_recpr_;

		PG pos_getter_;

		std::vector<Record> records_;

		::pbf::hash<cell_comp_t> cell_hasher_;
		std::unordered_map<size_t, std::vector<size_t>> hash2idxs_;
	};
} // namespace pbf

#endif /* spatial_hash_h */
