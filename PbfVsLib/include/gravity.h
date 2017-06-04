//
//  test_force.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef test_force_h
#define test_force_h

#include "typedefs.h"

namespace pbf {
    class ParticleSystem;
    
    class GravityEffect {
    public:
        GravityEffect();
        
        void Evaluate(float dt, ParticleSystem* ps);
        
    private:
        vec_t gravity_;
    };
} // namespace pbf

#endif /* test_force_h */
