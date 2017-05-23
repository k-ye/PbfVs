#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <stdio.h>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// CUDA 8.0, only for test now
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "include/config.h"
#include "include/constants.h"

#include "include/arcball_camera.h"
#include "include/gl_fix.h"
#include "include/point_drawer.h"
#include "include/renderer.h"
#include "include/shader_wrapper.h"

#include "include/aabb.h"
#include "include/boundary_constraint.h"
#include "include/gravity.h"
#include "include/particle_system.h"
#include "include/pbf_solver.h"
#include "include/spatial_hash.h"

////////////////////////////////////////////////////

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;

float world_size = 0.0f;
float delta_time = 0.0f;

// Camera instance
pbf::ArcballCamera camera;

// Particle System instance
pbf::ParticleSystem ps;

// PBF Solver instance
pbf::PbfSolver solver;

// SceneRender instance
pbf::SceneRenderer render;

////////////////////////////////////////////////////

// Configure the parameters of the world
void Configure(pbf::Config& config);

void InitParticles(const pbf::Config& config);

void InitDependencies();

////////////////////////////////////////////////////

// Callback function declarations
bool is_paused = false;
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);

bool left_btn_pressed = false;
void MouseCallback(GLFWwindow* window, double xpos, double ypos);

float max_arcball_radius = 100.0f;
void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

////////////////////////////////////////////////////

namespace {

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

} // namespace anonymous


// The MAIN function, from here we start the application and run the game loop
int main() {
	std::cout << "Starting GLFW context, OpenGL 3.3" << std::endl;
	// Init GLFW
	glfwInit();
	// Set all the required options for GLFW
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	GLFW_FORWARD_COMPATIBLE();

	// Create a GLFWwindow object that we can use for GLFW's functions
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "PBF", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	// Initialize PBF
	pbf::Config config;
	config.Load("Config/config.txt");
	Configure(config);

	InitParticles(config);

	InitDependencies();

	// Set the required callback functions
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCursorPosCallback(window, MouseCallback);
	glfwSetScrollCallback(window, ScrollCallback);

	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
	glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
	glewInit();

	// Define the viewport dimensions
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	render.InitShaders("Shaders/vertex.vert", "Shaders/fragment.frag");
	render.InitScene();

	// pbf::GravityEffect ge;
	// pbf::CubicBoundaryConstraint cbc;
	// cbc.set_boundary_size(world_size);

	// Game loop
	while (!glfwWindowShouldClose(window)) {
		// Check if any events have been activiated (key pressed, mouse moved etc.)
		// and call corresponding response functions
		glfwPollEvents();

		if (!is_paused) {
			// ge.Evaluate(delta_time, &ps);
			// cbc.Apply(&ps);
			solver.Update(delta_time);
		}
		render.Render();

		// Swap the screen buffers
		glfwSwapBuffers(window);
	}

	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();
	return 0;
}

////////////////////////////////////////////////////

void ConfigureCamera(const pbf::Config& config)
{
	// config camera
	camera.SetStageSize(WIDTH, HEIGHT);

	float radius = config.Get<float>(pbf::INIT_ARCBALL_RADIUS);
	camera.SetArcballRadius(radius);
	float sensitivity = 2.0f;
	config.GetOptional(pbf::CAMERA_SENSITIVITY, &sensitivity);
	camera.SetSensitivity(sensitivity);

	max_arcball_radius = config.Get<float>(pbf::MAX_ARCBALL_RADIUS);
}

void ConfigureSolver(const pbf::Config& config)
{
	pbf::PbfSolverConfig solver_config;

	solver_config.h = config.Get<float>(pbf::H_KERNEL);
	solver_config.mass = config.Get<float>(pbf::PARTICLE_MASS);
	solver_config.rho_0 = config.Get<float>(pbf::RHO_0);
	solver_config.epsilon = config.Get<float>(pbf::EPSILON);
	solver_config.num_iters = config.Get<unsigned>(pbf::NUM_ITERATIONS);
	solver_config.corr_delta_q_coeff = config.Get<float>(pbf::CORR_DELTA_Q_COEFF);
	solver_config.corr_k = config.Get<float>(pbf::CORR_K);
	solver_config.corr_n = config.Get<unsigned>(pbf::CORR_N);
	solver_config.vorticity_epsilon = config.Get<float>(pbf::VORTICITY_EPSILON);
	solver_config.xsph_c = config.Get<float>(pbf::XSPH_C);

	solver_config.world_size = config.Get<float>(pbf::WORLD_SIZE);
	solver_config.spatial_hash_cell_size = config.Get<float>(pbf::SH_CELL_SIZE);

	solver.Configure(solver_config);
}

void ConfigureRenderer(const pbf::Config& config)
{
	render.SetWorldSize(world_size);

	float fov = 45.0f;
	config.GetOptional(pbf::FOV, &fov);
	float aspect = (float)WIDTH / (float)HEIGHT;
	float near = 0.1f;
	config.GetOptional(pbf::PROJECTION_NEAR, &near);
	float far = config.Get<float>(pbf::PROJECTION_FAR);
	render.SetPespectiveProjection(fov, aspect, near, far);
}

void Configure(pbf::Config& config)
{
	world_size = config.Get<float>(pbf::WORLD_SIZE);
	delta_time = config.Get<float>(pbf::DELTA_TIME);

	ConfigureCamera(config);

	ConfigureSolver(config);

	ConfigureRenderer(config);
}

void InitParticles(const pbf::Config& config) {
	srand(time(nullptr));

	// float half_world_size = world_size * 0.5f;
	unsigned num_x = config.Get<unsigned>(pbf::NUM_PTCS_WIDTH);
	unsigned num_z = config.Get<unsigned>(pbf::NUM_PTCS_HEIGHT);
	unsigned num_y = config.Get<unsigned>(pbf::NUM_PTC_LAYERS);
	float interval = config.Get<float>(pbf::PARTICLE_INTERVAL);

	// glm::vec3 velocity{ 0.0f };

	float margin = (world_size - (num_x - 1) * interval) * 0.5f;

	for (unsigned y = 0; y < num_y; ++y) {
		for (unsigned z = 0; z < num_z; ++z) {
			for (unsigned x = 0; x < num_x; ++x) {
				float xf = margin + x * interval;
				float yf = world_size - margin - y * interval;
				float zf = margin + z * interval;
				const glm::vec3 pos{ xf, yf, zf };

				float vx = RandomFloat(-0.5f, 0.5f);
				float vy = RandomFloat(0.0f, 1.0f);
				float vz = RandomFloat(-0.5f, 0.5f);
				const glm::vec3 vel{ vx, vy, vz };

				ps.Add(pos, vel);
			}
		}
	}
}

void InitDependencies()
{
	solver.InitParticleSystems(&ps);

	render.SetCamera(&camera);
	render.SetParticleSystem(&ps);
}

////////////////////////////////////////////////////

// Is called whenever a key is pressed/released via GLFW
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		is_paused = !is_paused;
}

void MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	int action = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);

	if (action == GLFW_PRESS)
	{
		if (!left_btn_pressed)
		{
			std::cout << "mouse left button just pressed" << std::endl;
			left_btn_pressed = true;
			camera.OnMouseLeftClick(xpos, ypos);
		}
		else
		{
			std::cout << "mouse left button dragging" << std::endl;
			camera.OnMouseLeftDragging(xpos, ypos);
		}
	}
	else
	{
		if (left_btn_pressed)
		{
			left_btn_pressed = false;
			camera.OnMouseLeftRelease(xpos, ypos);
			std::cout << "mouse left button released" << std::endl;
		}
	}
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	float arcball_radius = camera.GetArcballRadius();
	arcball_radius += yoffset * 0.1f;
	std::cout << "scroll! yoffset: " << yoffset << ", radius: " << arcball_radius << std::endl;
	if (arcball_radius > 0 && arcball_radius < max_arcball_radius)
	{
		camera.SetArcballRadius(arcball_radius);
	}
}
