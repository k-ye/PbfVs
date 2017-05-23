#ifndef SH_POSITION_GETTER_H
#define SH_POSITION_GETTER_H

#include "particle_system.h"

namespace pbf 
{
	class PositionGetter
	{
	public:
		PositionGetter(ParticleSystem* ps = nullptr) : ps_(ps) { }

		point_t operator()(size_t i) const
		{
			return ps_->Get(i).position();
		}
	private:
		ParticleSystem* ps_;
	};
} // namespace pbf
#endif