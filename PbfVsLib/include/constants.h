#ifndef constants_h
#define constants_h

#include <string>

namespace pbf {
    
	////////////////////////////////////////////////////
	// All the key names used in the config file of this solver

	// World settings 
	const std::string WORLD_SIZE = "world_size";
	const std::string DELTA_TIME = "delta_time";

	// Camera
	const std::string FOV = "fov";
	const std::string PROJECTION_NEAR = "projection_near";
	const std::string PROJECTION_FAR = "projection_far";

	const std::string CAMERA_SENSITIVITY = "camera_sensitivity";
	const std::string INIT_ARCBALL_RADIUS = "init_arcball_radius";
	const std::string MAX_ARCBALL_RADIUS = "max_arcball_radius";

	// Particle system
	const std::string NUM_PTCS_WIDTH = "num_particles_width";
	const std::string NUM_PTCS_HEIGHT = "num_particles_height";
	const std::string NUM_PTC_LAYERS = "num_particle_layers";
	const std::string PARTICLE_INTERVAL = "particle_interval";

	// PBF
	const std::string H_KERNEL = "h";
	const std::string PARTICLE_MASS = "mass";
	const std::string RHO_0 = "rho_0";
	const std::string EPSILON = "epsilon";
	const std::string SH_CELL_SIZE = "sh_cell_size";
	const std::string NUM_ITERATIONS = "num_iterations";

	const std::string CORR_DELTA_Q_COEFF = "corr_delta_q_coeff";
	const std::string CORR_K = "corr_k";
	const std::string CORR_N = "corr_n";

	const std::string VORTICITY_EPSILON = "vorticity_epsilon";
	const std::string XSPH_C = "xsph_c";
} // namespace pbf
#endif /* constants_h */
