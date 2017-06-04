//
//  renderer.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/renderer.h"

#include "../include/arcball_camera.h"
#include "../include/shared_math.h"
#include "../include/point_drawer.h"
#include "../include/shader_wrapper.h"

#include <algorithm>
#include <unordered_set>

namespace pbf {
namespace {
	float Base_(float val) {
		if (val <= -0.75f) 
			return 0.0f;
		else if (val <= -0.25f) 
			return Interpolate(val, -0.75f, -0.25f, 0.0f, 1.0f);
		else if (val <= 0.25) 
			return 1.0f;
		else if (val <= 0.75f) 
			return Interpolate(val, 0.25f, 0.75f, 0.0f, 1.0f);
		return 0.0f;
	}

	float Red(float val) { return Base_(val - 0.5f); }
	float Green(float val) { return Base_(val); }
	float Blue(float val) { return Base_(val + 0.5f); }
} // namespace anonymous

    void SceneRenderer::SetWorldSize(float s) {
        world_sz_ = s;
        
        float half_sz = s * 0.5f;
        
        glm::mat4 model;
        model = glm::translate(model, glm::vec3(-half_sz, -half_sz, -half_sz));
        model_ = model;
    }
    
    void SceneRenderer::SetCamera(pbf::ArcballCamera* camera) {
        camera_ = camera;
    }
    
    void SceneRenderer::SetParticleSystem(pbf::ParticleSystem *ps) {
        ps_ = ps;
    }
    
    void SceneRenderer::SetPespectiveProjection(float fov, float wh_aspect, 
		float near, float far) {
        proj_ = glm::perspective(fov, wh_aspect, near, far);
    }
    
    void SceneRenderer::InitShaders(const char* vert_path, const char* frag_path) {
        using namespace pbf;
        // Create shaders
        // Shaders only need to be created once. Once they are
        // attached to a shader program, they can be deleted safely.
        GLShaderWrapper vert_shader(GL_VERTEX_SHADER, vert_path);
        GLShaderWrapper frag_shader(GL_FRAGMENT_SHADER, frag_path);
        
        // Link shaders
        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vert_shader.Get());
        glAttachShader(shader_program_, frag_shader.Get());
        glLinkProgram(shader_program_);
    }
    
    void SceneRenderer::SetVao_(GLuint vao, GLuint vbo, GLuint ebo) const {
        glBindVertexArray(vao);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        
        GLsizei stride = sizeof(GLfloat) * 6;
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (GLvoid*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (GLvoid*)(sizeof(GLfloat) * 3));
        glEnableVertexAttribArray(1);
        
        // Unbind VAO first, then VBO and EBO!
        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    
    void SceneRenderer::InitScene() {
        // Set up world cube vertex data
        world_cube_vertices_ = {
			// position          color
            0.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.2f,                  // 0, left, bottom, far
            world_sz_, 0.0f, 0.0f, 1.0f, 0.5f, 0.2f,             // 1, right, bottom, far
            world_sz_, 0.0f, world_sz_, 1.0f, 0.5f, 0.2f,        // 2, right, bottom, near
            0.0f, 0.0f, world_sz_, 1.0f, 0.5f, 0.2f,             // 3, left, bottom, near
            0.0f, world_sz_, 0.0f, 1.0f, 0.5f, 0.2f,             // 4, left, top, far
            world_sz_, world_sz_, 0.0f, 1.0f, 0.5f, 0.2f,        // 5, right, top, far
            world_sz_, world_sz_, world_sz_, 1.0f, 0.5f, 0.2f,   // 6, right, top, near
            0.0f, world_sz_, world_sz_, 1.0f, 0.5f, 0.2f,        // 7, left, top, near
        };
        
        world_cube_indices_ = {
            0, 1, 2,
            0, 2, 3,
            0, 4, 5,
            0, 5, 1,
            0, 3, 7,
            0, 7, 4
        };
        
		frame_vertices_ = {
			// position          color
			-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  // 0, x from
			-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // 1, y from
			-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // 2, z from
			+2.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  // 3, x to
			-1.0f, +2.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // 4, y to
			-1.0f, -1.0f, +2.0f, 0.0f, 1.0f, 0.0f,  // 5, z to
		};
		
		frame_indices_ = {
			0, 3,
			1, 4,
			2, 5
		};

        // set up world scene
        glGenVertexArrays(1, &world_vao_);
        glGenBuffers(1, &world_vbo_);
        glGenBuffers(1, &world_ebo_);
        
        glBindBuffer(GL_ARRAY_BUFFER, world_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * world_cube_vertices_.size(),
                     world_cube_vertices_.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, world_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * world_cube_indices_.size(),
                     world_cube_indices_.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        
        SetVao_(world_vao_, world_vbo_, world_ebo_);
		// set up frame
		glGenVertexArrays(1, &frame_vao_);
		glGenBuffers(1, &frame_vbo_);
		glGenBuffers(1, &frame_ebo_);

		glBindBuffer(GL_ARRAY_BUFFER, frame_vbo_);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * frame_vertices_.size(),
			frame_vertices_.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, frame_ebo_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * frame_indices_.size(),
			frame_indices_.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		SetVao_(frame_vao_, frame_vbo_, frame_ebo_);
        // set up particles
        for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
            auto ptc = ps_->Get(p_i);
            AddPointToDraw(ptc.position(), &particle_vertices_, &particle_indices_);
        }
        
        glGenVertexArrays(1, &particles_vao_);
        glGenBuffers(1, &particles_vbo_);
        glGenBuffers(1, &particles_ebo_);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particles_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * particle_indices_.size(),
                     particle_indices_.data(), GL_STATIC_DRAW);
        
        SetVao_(particles_vao_, particles_vbo_, particles_ebo_);
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    
    void SceneRenderer::Render() {
        // Render
        // Clear the colorbuffer
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(shader_program_);
        
        GLuint model_loc = glGetUniformLocation(shader_program_, "model");
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(model_));
        
        glm::mat4 view = camera_->GetViewMatrix();
        GLuint view_loc = glGetUniformLocation(shader_program_, "view");
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));
        
        GLuint proj_loc = glGetUniformLocation(shader_program_, "proj");
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm::value_ptr(proj_));

		// draw the world bounding box
        glBindVertexArray(world_vao_);
        glDrawElements(GL_TRIANGLES, (int)world_cube_indices_.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

		// draw the xyz frame
		glBindVertexArray(frame_vao_);
		glDrawElements(GL_LINES, (int)frame_indices_.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// draw particles
        // https://www.gamedev.net/topic/597387-vao-is-it-necessary-to-redo-setup-each-time-buffer-data-changes/ 
        for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
			const auto pos = ps_->Get(p_i).position();
			float col_in = Interpolate(pos.y + world_sz_ * 0.5f, 0.0f, world_sz_, 0.0f, 1.0f);
			col_in = std::max(std::min(col_in, 1.0f), 0.0f);
			const glm::vec3 color{ 1.0f, col_in, 1.0f - col_in };
			ChangePointToDraw(pos, p_i, &particle_vertices_, color);
        }
        
        glBindBuffer(GL_ARRAY_BUFFER, particles_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * particle_vertices_.size(),
                     particle_vertices_.data(), GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(particles_vao_);
        glDrawElements(GL_TRIANGLES, (int)particle_indices_.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glUseProgram(0); 
    }
} // namespace pbf
