//
//  boundary_constraint.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/boundary_constraint.h"
#include "../include/particle_system.h"

namespace pbf
{
    void CubicBoundaryConstraint::set_boundary_size(float s)
    {
        boundary_size_ = s;
    }
    
    void CubicBoundaryConstraint::Apply(pbf::ParticleSystem *ps)
    {
        for (size_t i = 0; i < ps->NumParticles(); ++i)
        {
            auto ptc = ps->Get(i);
            auto pos = ptc.position();
            auto vel = ptc.velocity();
            
            for (int c = 0; c < 3; ++c)
            {
                bool out1 = (pos[c] < 0.0f);
                bool out2 = (pos[c] > boundary_size_);
                if (out1 || out2)
                {
                    vel[c] = -vel[c];
                    
                    // to avoid oscillation
                    if (out1)
                        pos[c] = FLT_EPSILON;
                    else if (out2)
                        pos[c] = boundary_size_ - FLT_EPSILON;
                }
            }
            
            ptc.set_position(pos);
            ptc.set_velocity(vel);
        }
    }
} // namespace pbf
