//
//  point_drawer.h
//  PBF
//
//  Created by Ye Kuang on 3/30/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#ifndef point_drawer_h
#define point_drawer_h

#include "glm_headers.h"

#include <GL/glew.h>
#include <vector>

namespace pbf {
    ////////////////////////////////////////////////////
    // These functions draw a point as a octahedron.
    
    void AddPointToDraw(const glm::vec3& pt, std::vector<GLfloat>* vertices,
		std::vector<GLuint>* indices, 
		const glm::vec3& color = glm::vec3{1.0f, 0.5f, 0.2f},
		float size = 0.5f);
    
    void ChangePointToDraw(const glm::vec3& pt, size_t index, 
		std::vector<GLfloat>* vertices, 
		const glm::vec3& color = glm::vec3{1.0f, 0.5f, 0.2f},
		float size = 0.5f);
} // namespace pbf

#endif /* point_drawer_h */
