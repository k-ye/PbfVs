//
//  gravity.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/gravity.h"

#include "../include/particle_system.h"

namespace pbf
{
    GravityEffect::GravityEffect() : gravity_(0.0f, -9.8f, 0.0f) { }
    
    void GravityEffect::Evaluate(float dt, ParticleSystem* ps)
    {
        for (size_t i = 0; i < ps->NumParticles(); ++i)
        {
            auto ptc = ps->Get(i);
            auto vel = ptc.velocity();
            auto pos = ptc.position();
            
            vel += gravity_ * dt;
            pos += vel * dt;
            ptc.set_position(pos);
            ptc.set_velocity(vel);
        }
    }
} // namespace pbf
