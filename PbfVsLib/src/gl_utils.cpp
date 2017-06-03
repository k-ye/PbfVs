#include "../include/gl_utils.h"

#include <iostream>

namespace pbf {
    void CHECK_SHADER_COMPILE_STATUS(GLint shader) {
        GLint success;
        GLchar info_log[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        
        if(!success) {
            glGetShaderInfoLog(shader, 512, NULL, info_log);
            std::cerr << "ERROR: Shader compilation failed\n" << info_log << std::endl;
        }
        else {
            std::cout << "Shader compilation succeeded!" << std::endl;
        }
    }
    
} // namespace pbf
