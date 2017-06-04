//
//  point_drawer.cpp
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/point_drawer.h"

namespace pbf {
namespace {
    // One octahedron has 6 vertices and 8 faces. Each vertex stores
    // its (position, color), which requires 6 GLfloats.Therefore,
    // each vertex is associated with 6 * 6 = 26 GLfloats
    // and 8 * 3 = 24 GLuints.
    enum { VERTEX_DATA_PER_POINT = 36u, INDEX_DATA_PER_POINT = 24u };
    
    typedef glm::vec3 vec3;
    
    std::vector<vec3> GetVertexOffsets_(float size) {
        std::vector<vec3> result = {
            vec3{0.0f, size, 0.0f},     // 0 top
            vec3{-size, 0.0f, 0.0f},    // 1 left
            vec3{0.0f, 0.0f, size},     // 2 front
            vec3{size, 0.0f, 0.0f},     // 3 right
            vec3{0.0f, 0.0f, -size},    // 4 back
            vec3{0.0f, -size, 0.0f}     // 5 bottom
        };
        
        return result;
    }
} // namespace detail_
    
    void AddPointToDraw(const glm::vec3& pt,
                        std::vector<GLfloat>* vertices,
                        std::vector<GLuint>* indices,
                        const glm::vec3& color,
                        float size) {
        
        unsigned vsz = (unsigned)vertices->size();
        assert(vsz % 6 == 0);
        unsigned base_index = vsz / 6;
        
        auto AddVertex = [vertices, &color](const vec3& v) {
            vertices->push_back(v[0]);
            vertices->push_back(v[1]);
            vertices->push_back(v[2]);
            
            vertices->push_back(color[0]);
            vertices->push_back(color[1]);
            vertices->push_back(color[2]);
        };
        
        auto AddIndices = [=](unsigned a, unsigned b, unsigned c) {
            indices->push_back(a);
            indices->push_back(b);
            indices->push_back(c);
        };
        
        
        for (const auto& offs : GetVertexOffsets_(size))
            AddVertex(pt + offs);
        
        AddIndices(base_index, base_index + 1, base_index + 2);
        AddIndices(base_index, base_index + 2, base_index + 3);
        AddIndices(base_index, base_index + 3, base_index + 4);
        AddIndices(base_index, base_index + 4, base_index + 1);
        AddIndices(base_index + 5, base_index + 1, base_index + 4);
        AddIndices(base_index + 5, base_index + 4, base_index + 3);
        AddIndices(base_index + 5, base_index + 3, base_index + 2);
        AddIndices(base_index + 5, base_index + 2, base_index + 1);
    }
    
    
    void ChangePointToDraw(const glm::vec3& pt, size_t index,
		std::vector<GLfloat>* vertices, const glm::vec3& color, float size) {
        unsigned vidx = index * VERTEX_DATA_PER_POINT;
        
        auto ChangeVertex = [vertices, &color](const vec3& v, unsigned vi_begin) {
            (*vertices)[vi_begin]     = v[0];
            (*vertices)[vi_begin + 1] = v[1];
            (*vertices)[vi_begin + 2] = v[2];
            
            (*vertices)[vi_begin + 3] = color[0];
            (*vertices)[vi_begin + 4] = color[1];
            (*vertices)[vi_begin + 5] = color[2];
        };
        
        for (const auto& offs : GetVertexOffsets_(size)) {
            ChangeVertex(pt + offs, vidx);
            vidx += 6; // point to the start of the next vertex
        }
    }
} // namespace pbf
