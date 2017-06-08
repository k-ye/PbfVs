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

    void SceneRenderer::SetWorldSize(const vec_t& s) {
        world_sz_x_ = s.x;
        world_sz_y_ = s.y;
        world_sz_z_ = s.z;
                
        glm::mat4 model;
        model = glm::translate(model, glm::vec3(-s.x * 0.5f, -s.y * 0.5f, -s.z * 0.5f));
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
        // using namespace pbf;
        shader_program_.Init(vert_path, frag_path);
        // // Link shaders
        // shader_program_ = glCreateProgram();
        // glAttachShader(shader_program_, vert_shader.Get());
        // glAttachShader(shader_program_, frag_shader.Get());
        // glLinkProgram(shader_program_);
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
        // TODO: huge function, split

        // set up boundaries
        PrepareBoundaryBuffers_();

        auto StoreBoundaryIndices = [this](size_t i) {
            GLuint vertex_begin = i * 4;
            size_t vidx_begin = i * 6;
            boundary_indices_[vidx_begin + 0] = vertex_begin;
            boundary_indices_[vidx_begin + 1] = vertex_begin + 1;
            boundary_indices_[vidx_begin + 2] = vertex_begin + 2;
            boundary_indices_[vidx_begin + 3] = vertex_begin;
            boundary_indices_[vidx_begin + 4] = vertex_begin + 2;
            boundary_indices_[vidx_begin + 5] = vertex_begin + 3;
        };

        for (size_t i = 0; i < boundary_records_.size(); ++i) {
            // UpdateBoundaryAt_(i);
            StoreBoundaryIndices(i);
        }

        glGenVertexArrays(1, &boundaries_vao_);
        glGenBuffers(1, &boundaries_vbo_);
        glGenBuffers(1, &boundaries_ebo_);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundaries_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * boundary_indices_.size(),
                     boundary_indices_.data(), GL_STATIC_DRAW);
        SetVao_(boundaries_vao_, boundaries_vbo_, boundaries_ebo_);

		// set up frame
		frame_vertices_ = {
			// position          color
			-1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  // 0, x from
			-1.0f, -1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // 1, y from
			-1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // 2, z from
			+2.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f,  // 3, x to
			-1.0f, +2.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // 4, y to
			-1.0f, -1.0f, +2.0f, 0.0f, 0.0f, 1.0f,  // 5, z to
		};
		
		frame_indices_ = {
			0, 3,
			1, 4,
			2, 5
		};
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
    
    void SceneRenderer::PrepareBoundaryBuffers_() {
        const int num_boundaries = boundary_records_.size();
        boundary_vertices_.resize(4 * 6 * num_boundaries, 0.0f);
        boundary_indices_.resize(6 * num_boundaries, 0);
    }

    void SceneRenderer::UpdateBoundaryAt_(size_t i) {
        // if (i == 1) return;

        const auto& brec = boundary_records_[i];
        const auto boundary = const_cast<const BoundaryConstraintBase*>(boundary_constraint_)->Get(i);
        const vec_t b_anchor = boundary.position;
        const vec_t b_normal = boundary.normal;
        auto Transform = [=]() -> glm::mat4 {
            const vec_t ref_normal{ 1.0f, 0.0f, 0.0f };
            const auto ref_b_normal_dot = glm::dot(ref_normal, b_normal);
            float rot_angle = 0.0f;
            vec_t rot_axis{ 0.0f };
            glm::mat4 result;
            if (glm::abs(1.0f - ref_b_normal_dot) <= kFloatEpsilon) {
                // ref_normal and b_normal is aligned
                return result;
            }
            else if (glm::abs(-1.0f - ref_b_normal_dot) <= kFloatEpsilon) {
                // ref_normal and b_normal points to the opposite direction
                rot_angle = glm::acos(-1.0f);
                // pick up +y dir that is perpendicular to |ref_normal|
                rot_axis = vec_t{ 0.0f, 1.0f, 0.0f };
            }
            else {
                rot_angle = glm::acos(dot(ref_normal, b_normal));
                rot_axis = glm::normalize(glm::cross(ref_normal, b_normal));
            }
            result = glm::rotate(result, rot_angle, rot_axis);
            return result;
        };
        auto tran_m = Transform();
        // glm::vec4 to glm::vec3 discards the last component
        vec_t dir1 = (tran_m * glm::vec4{ 0.0f, 0.0f, brec.v1_len, 0.0f });
        vec_t dir2 = (tran_m * glm::vec4{ 0.0f, brec.v2_len, 0.0f, 0.0f });

        point_t v0 = b_anchor;
        point_t v1 = b_anchor + dir1;
        point_t v2 = b_anchor + dir1 + dir2;
        point_t v3 = b_anchor + dir2;
        
        size_t vertex_begin = i * 6 * 4;
        boundary_vertices_[vertex_begin + 0] = v0.x;
        boundary_vertices_[vertex_begin + 1] = v0.y;
        boundary_vertices_[vertex_begin + 2] = v0.z;
        boundary_vertices_[vertex_begin + 3] = 1.0f;
        boundary_vertices_[vertex_begin + 4] = 0.5f;
        boundary_vertices_[vertex_begin + 5] = 0.2f;
        
        boundary_vertices_[vertex_begin + 6] = v1.x;
        boundary_vertices_[vertex_begin + 7] = v1.y;
        boundary_vertices_[vertex_begin + 8] = v1.z;
        boundary_vertices_[vertex_begin + 9] = 1.0f;
        boundary_vertices_[vertex_begin + 10] = 0.5f;
        boundary_vertices_[vertex_begin + 11] = 0.2f;
        
        boundary_vertices_[vertex_begin + 12] = v2.x;
        boundary_vertices_[vertex_begin + 13] = v2.y;
        boundary_vertices_[vertex_begin + 14] = v2.z;
        boundary_vertices_[vertex_begin + 15] = 1.0f;
        boundary_vertices_[vertex_begin + 16] = 0.5f;
        boundary_vertices_[vertex_begin + 17] = 0.2f;
        
        boundary_vertices_[vertex_begin + 18] = v3.x;
        boundary_vertices_[vertex_begin + 19] = v3.y;
        boundary_vertices_[vertex_begin + 20] = v3.z;
        boundary_vertices_[vertex_begin + 21] = 1.0f;
        boundary_vertices_[vertex_begin + 22] = 0.5f;
        boundary_vertices_[vertex_begin + 23] = 0.2f;
    }
    
    void SceneRenderer::Render() {
        // Render
        // Clear the colorbuffer
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        shader_program_.Use();
        
        GLuint model_loc = glGetUniformLocation(shader_program_.Get(), "model");
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm::value_ptr(model_));
        
        glm::mat4 view = camera_->GetViewMatrix();
        GLuint view_loc = glGetUniformLocation(shader_program_.Get(), "view");
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));
        
        GLuint proj_loc = glGetUniformLocation(shader_program_.Get(), "proj");
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm::value_ptr(proj_));
        
        // draw the boundaries
        for (size_t i = 0; i < boundary_records_.size(); ++i) {
            UpdateBoundaryAt_(i);
        }
        glBindBuffer(GL_ARRAY_BUFFER, boundaries_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * boundary_vertices_.size(),
                     boundary_vertices_.data(), GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        glBindVertexArray(boundaries_vao_);
        glDrawElements(GL_TRIANGLES, (int)boundary_indices_.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

		// draw the xyz frame
		glBindVertexArray(frame_vao_);
		glDrawElements(GL_LINES, (int)frame_indices_.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// draw the particles
        // https://www.gamedev.net/topic/597387-vao-is-it-necessary-to-redo-setup-each-time-buffer-data-changes/ 
        for (size_t p_i = 0; p_i < ps_->NumParticles(); ++p_i) {
			const auto pos = ps_->Get(p_i).position();
			float col_in = Interpolate(pos.y + world_sz_y_ * 0.5f, 0.0f, world_sz_y_, 0.0f, 1.0f);
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

        shader_program_.Unbind();
    }
} // namespace pbf
