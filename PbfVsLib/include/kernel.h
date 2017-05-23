//
//  kernel.h
//  PBF
//
//  Created by Ye Kuang on 4/1/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef kernel_h
#define kernel_h

#include "basic.h"

namespace pbf
{

#define WKERNEL_DBG_MODE

	class WKernel
	{
	public:
		WKernel()
			: h_(0.0f), h_sqr_(0.0f), h_cube_(0.0f), h_inited_(false)
			,poly6_factor_(315.0f / 64.0f / glm::pi<float>())
			, spiky_grad_factor_(-45.0f / glm::pi<float>())
		{ }

		void set_h(float h)
		{
#ifdef WKERNEL_DBG_MODE 
			if (h_inited_) {
				throw "WKernel: cannot init h multiple times";
			}
#endif
			h_ = h;
			h_sqr_ = h * h;
			h_cube_ = h_sqr_ * h;
			h_inited_ = true;
		}

		// Poly6 function value for scalar input 
		float Evaluate(float s) const {
			CheckHInited();
			
			if (s < 0.0f || s >= h_)
				return 0.0f;

			float x = (h_sqr_ - s * s) / h_cube_;
			float result = poly6_factor_ * x * x * x;

			return result;
		}

		// Poly6 function value for vector input. 
		float Evaluate(const vec_t& r) const {
			float r_len = glm::length(r);
			return Evaluate(r_len);
		}

		// Spiky function gradient
		vec_t Gradient(const vec_t& r) const {
			CheckHInited();

			float r_len = glm::length(r);
			if (r_len <= 0.0f || r_len >= h_)
				return vec_t{ 0.0f };

			float x = (h_ - r_len) / h_cube_;
			float g_factor = spiky_grad_factor_ * x * x; 
			vec_t result = glm::normalize(r) * g_factor;
			return result;
		}

		vec_t WViscosity(const vec_t& r) const {
			CheckHInited();

			float r_len = glm::length(r);
			float r_sqr = r_len * r_len;
			if (r_len <= 0.0f || r_len >= h_)
				return vec_t{ 0.0f };

			float coeff = (-0.5f * (r_len * r_len * r_len)) / h_cube_;
			coeff += (r_sqr / h_sqr_);
			coeff += ((0.5f * h_) / r_len) - 1.0f;
			return coeff * r;
		}
	private:
		inline void CheckHInited() const {
#ifdef WKERNEL_DBG_MODE
			if (!h_inited_) {
				throw "WKernel: Cannot evaluate since h is not inited yet";
			}
#endif
		}

	private:
		float h_;
		float h_sqr_;
		float h_cube_;
		bool h_inited_;

		float poly6_factor_;
		float spiky_grad_factor_;
	};

} // namespace pbf

#endif /* kernel_h */
