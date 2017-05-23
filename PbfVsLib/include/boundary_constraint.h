//
//  boundary_constraint.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef boundary_constraint_h
#define boundary_constraint_h

#include "basic.h"

namespace pbf
{
    class ParticleSystem;
    
    class CubicBoundaryConstraint
    {
    public:
        void set_boundary_size(float s);
        
        void Apply(ParticleSystem* ps);
        
    private:
        
        float boundary_size_;
    };
} // namespace pbf

#endif /* boundary_constraint_h */
