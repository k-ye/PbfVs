#include <iostream>
#include <stdio.h>
#include <stdlib.h> // srand, rand
#include <time.h>   // time
#include <vector>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// CUDA 8.0, only for test now
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// PBF
#include "include/arcball_camera.h"
#include "include/boundary_gpu.h"
#include "include/config.h"
#include "include/constants.h"
#include "include/gl_fix.h"
#include "include/obj_model.h"
#include "include/obj_models_helpers.h"
#include "include/particle_system.h"
#include "include/pbf_solver.h"
#include "include/pbf_solver_gpu.h"
#include "include/point_drawer.h"
#include "include/renderer.h"
#include "include/shader_wrapper.h"
#include "include/shared_math.h"
#include "include/spatial_hash.h"

////////////////////////////////////////////////////

// TODO(k-ye): These global variables should be cleaned up

// Window dimensions
// const GLuint WIDTH = 1024, HEIGHT = 768;
const GLuint WIDTH = 768;
const GLuint HEIGHT = 1280;

float delta_time = 0.0f;
glm::vec3 world_size_dim{0.0f};

// Camera instance
pbf::ArcballCamera camera;

// Particle System instance
pbf::ParticleSystem ps;

pbf::BoundaryConstraintGpu boundary_constraint;

// PBF Solver instance
// pbf::PbfSolver solver;
pbf::PbfSolverGpu solver;

// SceneRender instance
pbf::SceneRenderer render;

////////////////////////////////////////////////////

// Configure the parameters of the world
void Configure(pbf::Config &config);

void InitParticles(const pbf::Config &config);

void InitDependencies();

////////////////////////////////////////////////////

// Callback function declarations
bool is_paused = false;
void KeyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mode);

bool left_btn_pressed = false;
void MouseCallback(GLFWwindow *window, double xpos, double ypos);

float max_arcball_radius = 100.0f;
void ScrollCallback(GLFWwindow *window, double xoffset, double yoffset);

////////////////////////////////////////////////////

// A class that moves the x hi boundary back and forth
class MoveXBoundaryDriver {
public:
  MoveXBoundaryDriver(pbf::BoundaryConstraintBase *bc) : bc_(bc) {}

  void Configure(const pbf::Config &config) {
    x_hi_index_ = 1;
    x_vel_ = 8.0f;
    const float world_size_x = config.Get<float>(pbf::WORLD_SIZE_X);
    x_lo_ = world_size_x * 0.6f;
    x_hi_ = world_size_x - 0.5f;
  }

  void Update(float dt) {
    auto *bp = bc_->Get(x_hi_index_);
    bp->position.x += (bp->velocity.x * dt);
    if (bp->position.x < x_lo_) {
      bp->position.x = x_lo_ + kFloatEpsilon;
      bp->velocity.x = x_vel_;
    } else if (bp->position.x > x_hi_) {
      bp->position.x = x_hi_ - kFloatEpsilon;
      bp->velocity.x = -x_vel_;
    }
  }

private:
  pbf::BoundaryConstraintBase *bc_;
  float x_vel_;
  float x_lo_;
  float x_hi_;
  size_t x_hi_index_;
};
////////////////////////////////////////////////////
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
  GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "PBF", nullptr, nullptr);
  glfwMakeContextCurrent(window);

  // Initialize PBF
  pbf::Config config;
  config.Load("Config/config.txt");
  Configure(config);

  InitParticles(config);
  // Once loaded, |obj_models| should never change its size.
  std::vector<pbf::ObjModel> obj_models =
      pbf::LoadModelsFromConfigFile("Config/model_defs.txt");
  {
    float interval = config.Get<float>(pbf::PARTICLE_INTERVAL);
    std::vector<pbf::point_t> obj_models_points =
        pbf::FillPointsInObjModels(obj_models, world_size_dim, interval);
    std::cout << "found " << obj_models_points.size() << " particles from the object models" << std::endl;
    for (const auto& pt : obj_models_points) {
        glm::vec3 vel;
        vel.x = 0.0f; // pbf::GenRandom(-0.05f, 0.05f);
        vel.y = pbf::GenRandom(-0.05f, 0.05f);
        vel.z = 0.0f; // pbf::GenRandom(-0.05f, 0.05f);
        ps.Add(pt, vel);;
    }
  }

  InitDependencies();

  MoveXBoundaryDriver boundary_driver{&boundary_constraint};
  boundary_driver.Configure(config);

  // Set the required callback functions
  glfwSetKeyCallback(window, KeyCallback);
  glfwSetCursorPosCallback(window, MouseCallback);
  glfwSetScrollCallback(window, ScrollCallback);

  // Set this to true so GLEW knows to use a modern approach to retrieving
  // function pointers and extensions
  glewExperimental = GL_TRUE;
  // Initialize GLEW to setup the OpenGL Function pointers
  glewInit();

  // Define the viewport dimensions
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  render.InitShaders("Shaders/vertex.vert", "Shaders/fragment.frag");
  render.InitSpriteShaders("Shaders/sprite_vertex.vert",
                           "Shaders/sprite_fragment.frag");
  for (const pbf::ObjModel &obj_model : obj_models) {
    // render.RegisterObjModel(&obj_model);
  }
  render.InitScene();

  is_paused = true;
  // Game loop
  while (!glfwWindowShouldClose(window)) {
    // Check if any events have been activiated (key pressed, mouse moved etc.)
    // and call corresponding response functions
    glfwPollEvents();

    if (!is_paused) {
      // boundary_driver.Update(delta_time);
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

void ConfigureCamera(const pbf::Config &config) {
  // config camera
  camera.SetStageSize(WIDTH, HEIGHT);

  float radius = config.Get<float>(pbf::INIT_ARCBALL_RADIUS);
  camera.SetArcballRadius(radius);
  float sensitivity = 2.0f;
  config.GetOptional(pbf::CAMERA_SENSITIVITY, &sensitivity);
  camera.SetSensitivity(sensitivity);

  max_arcball_radius = config.Get<float>(pbf::MAX_ARCBALL_RADIUS);
}

void ConfigureBoundaryConstraint(const pbf::Config &config) {
  using pbf::vec_t;

  const float world_size_x = world_size_dim.x;
  const float world_size_y = world_size_dim.y;
  const float world_size_z = world_size_dim.z;
  pbf::BoundaryPlane bp;
  // X lo
  bp.position = vec_t{0.0f, 0.0f, 0.0f};
  bp.velocity = vec_t{0.0f};
  bp.normal = vec_t{1.0f, 0.0f, 0.0f};
  boundary_constraint.Add(bp);
  // X hi
  bp.position = vec_t{world_size_x, 0.0f, world_size_z};
  bp.velocity = vec_t{0.0f};
  bp.normal = vec_t{-1.0f, 0.0f, 0.0f};
  boundary_constraint.Add(bp);
  // Z lo
  bp.position = vec_t{world_size_x, 0.0f, 0.0f};
  bp.velocity = vec_t{0.0f};
  bp.normal = vec_t{0.0f, 0.0f, 1.0f};
  boundary_constraint.Add(bp);
  // Z hi
  bp.position = vec_t{0.0f, 0.0f, world_size_z};
  bp.velocity = vec_t{0.0f};
  bp.normal = vec_t{0.0f, 0.0f, -1.0f};
  boundary_constraint.Add(bp);
  // Y lo
  bp.position = vec_t{world_size_x, 0.0f, 0.0f};
  bp.velocity = vec_t{0.0f};
  bp.normal = vec_t{0.0f, 1.0f, 0.0f};
  boundary_constraint.Add(bp);
  // No Y hi, top not covered
}

void ConfigureSolver(const pbf::Config &config) {
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

  solver_config.world_size_x = world_size_dim.x;
  solver_config.world_size_y = world_size_dim.y;
  solver_config.world_size_z = world_size_dim.z;
  solver_config.spatial_hash_cell_size = config.Get<float>(pbf::SH_CELL_SIZE);

  solver.Configure(solver_config);
}

void ConfigureRenderer(const pbf::Config &config) {
  render.SetWorldSize(world_size_dim);

  float fov = 45.0f;
  config.GetOptional(pbf::FOV, &fov);
  float aspect = (float)WIDTH / (float)HEIGHT;
  float near = 0.1f;
  config.GetOptional(pbf::PROJECTION_NEAR, &near);
  float far = config.Get<float>(pbf::PROJECTION_FAR);
  render.SetPespectiveProjection(fov, aspect, near, far);
}

void Configure(pbf::Config &config) {
  delta_time = config.Get<float>(pbf::DELTA_TIME);
  float world_size_x = config.Get<float>(pbf::WORLD_SIZE_X);
  float world_size_y = config.Get<float>(pbf::WORLD_SIZE_Y);
  float world_size_z = config.Get<float>(pbf::WORLD_SIZE_Z);
  world_size_dim = {world_size_x, world_size_y, world_size_z};

  ConfigureCamera(config);
  ConfigureBoundaryConstraint(config);
  ConfigureSolver(config);
  ConfigureRenderer(config);
}

void InitParticles(const pbf::Config &config) {
  srand(time(nullptr));

  unsigned num_x = config.Get<unsigned>(pbf::NUM_PTCS_WIDTH);
  unsigned num_z = config.Get<unsigned>(pbf::NUM_PTCS_HEIGHT);
  unsigned num_y = config.Get<unsigned>(pbf::NUM_PTC_LAYERS);
  float world_size_x = config.Get<float>(pbf::WORLD_SIZE_X);
  float world_size_y = config.Get<float>(pbf::WORLD_SIZE_Y);
  float world_size_z = config.Get<float>(pbf::WORLD_SIZE_Z);
  float interval = config.Get<float>(pbf::PARTICLE_INTERVAL);

  auto ComputeMargin = [=](float world_sz_dim, unsigned num_dim) -> float {
    return (world_sz_dim - ((num_dim - 1) * interval)) * 0.5f;
  };

  // float margin_y = ComputeMargin(world_size_y, num_y);
  float margin_y = interval * 0.5;
  float margin_z = ComputeMargin(world_size_z, num_z);
  float margin_x = ComputeMargin(world_size_x, num_x);
  for (unsigned y = 0; y < num_y; ++y) {
    for (unsigned z = 0; z < num_z; ++z) {
      for (unsigned x = 0; x < num_x; ++x) {
        float xf = margin_x + x * interval;
        // float yf = world_size_y - margin_y - y * interval;
        float yf = margin_y + y * interval;
        float zf = margin_z + z * interval;
        const glm::vec3 pos{xf, yf, zf};

        float vx = pbf::GenRandom(-0.5f, 0.5f);
        float vy = pbf::GenRandom(0.0f, 1.0f);
        float vz = pbf::GenRandom(-0.5f, 0.5f);
        const glm::vec3 vel{vx, vy, vz};

        ps.Add(pos, vel);
      }
    }
  }
}

void InitDependencies() {
  solver.InitParticleSystems(&ps);
  solver.SetBoundaryConstraint(&boundary_constraint);

  render.SetCamera(&camera);
  render.SetParticleSystem(&ps);
  render.boundary_constraint_ = &boundary_constraint;
  for (size_t i = 0; i < boundary_constraint.NumBoundaries(); ++i) {
    pbf::SceneRenderer::BoundaryRecord brec;
    brec.index = i;
    if (i == 0 || i == 1) {
      brec.v1_len = world_size_dim.z;
      brec.v2_len = world_size_dim.y;
    } else if (i == 2 || i == 3) {
      brec.v1_len = world_size_dim.x;
      brec.v2_len = world_size_dim.y;
    } else {
      brec.v1_len = world_size_dim.z;
      brec.v2_len = world_size_dim.x;
    }
    render.boundary_records_.push_back(brec);
  }
}

////////////////////////////////////////////////////

// Is called whenever a key is pressed/released via GLFW
void KeyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mode) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    is_paused = !is_paused;
}

void MouseCallback(GLFWwindow *window, double xpos, double ypos) {
  int action = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);

  if (action == GLFW_PRESS) {
    if (!left_btn_pressed) {
      std::cout << "mouse left button just pressed" << std::endl;
      left_btn_pressed = true;
      camera.OnMouseLeftClick(xpos, ypos);
    } else {
      std::cout << "mouse left button dragging" << std::endl;
      camera.OnMouseLeftDragging(xpos, ypos);
    }
  } else {
    if (left_btn_pressed) {
      left_btn_pressed = false;
      camera.OnMouseLeftRelease(xpos, ypos);
      std::cout << "mouse left button released" << std::endl;
    }
  }
}

void ScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
  float arcball_radius = camera.GetArcballRadius();
  arcball_radius += yoffset * 0.25f;
  std::cout << "scroll! yoffset: " << yoffset << ", radius: " << arcball_radius
            << std::endl;
  if (arcball_radius > 0 && arcball_radius < max_arcball_radius) {
    camera.SetArcballRadius(arcball_radius);
  }
}
