//
//  basic.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef basic_h
#define basic_h

#include "glm_headers.h"

namespace pbf
{
    typedef glm::vec3 point_t;
    typedef glm::vec3 vec_t;
	const float kFloatEpsilon = 1e-5;
	
	template <typename T>
	T Interpolate(T val, T vmin, T vmax, T rmin, T rmax) {
		return (val - vmin) * (rmax - rmin) / (vmax - vmin) + rmin;
	}
}
#endif /* basic_h */
