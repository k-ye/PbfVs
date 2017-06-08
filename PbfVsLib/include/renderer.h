#ifndef renderer_h
#define renderer_h

#include <vector>

#include "glm_headers.h"
#include "boundary_base.h"
#include "particle_system.h"
#include "shader_program.h"

namespace pbf {
    class ArcballCamera;
    class ParticleSystem;
    
    class SceneRenderer {
    public:
        SceneRenderer() = default;
        SceneRenderer(const SceneRenderer&) = delete;
        SceneRenderer& operator=(const SceneRenderer&) = delete;
        
        void SetWorldSize(const vec_t& s);
        
        void SetCamera(pbf::ArcballCamera* camera);
        
        void SetParticleSystem(pbf::ParticleSystem* ps);
       
        void SetPespectiveProjection(float fov, float wh_aspect, float near, float far);
        
        void InitShaders(const char* vert_path, const char* frag_path);
        
        void InitScene();
        
        void Render();
        
        void SetVao_(GLuint vao, GLuint vbo, GLuint ebo) const;
        
        void PrepareBoundaryBuffers_();

        void UpdateBoundaryAt_(size_t i);
        
    private:
        
        pbf::ArcballCamera* camera_;
        pbf::ParticleSystem* ps_;
        
        // OpenGL transformation matrices
        glm::mat4 model_;
        glm::mat4 proj_;
        
        // GLuint shader_program_;
        ShaderProgram shader_program_;

        // *world* is a cube that defines the boundary of the PBF.
        GLfloat world_sz_x_;
        GLfloat world_sz_y_;
        GLfloat world_sz_z_;
        // world boundary
        GLuint boundaries_vao_;
        GLuint boundaries_vbo_;
        GLuint boundaries_ebo_;
        
        std::vector<GLfloat> boundary_vertices_;
        std::vector<GLuint> boundary_indices_;
        
        // xyz frame (coordinate) 
		GLuint frame_vao_;
        GLuint frame_vbo_;
        GLuint frame_ebo_;
        
        std::vector<GLfloat> frame_vertices_;
        std::vector<GLuint> frame_indices_;
        // particles
        GLuint particles_vao_;
        GLuint particles_vbo_;
        GLuint particles_ebo_;
        
        std::vector<GLfloat> particle_vertices_;
        std::vector<GLuint> particle_indices_;

    public:
        // public for quick implementation
        struct BoundaryRecord {
            size_t index;
            float v1_len;
            float v2_len;
        };
        std::vector<BoundaryRecord> boundary_records_;
        BoundaryConstraintBase* boundary_constraint_;
    };
} // namespace pbf

#endif // renderer_h
