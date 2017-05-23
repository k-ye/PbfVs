//
//  shader_wrapper.cpp
//  PBF
//
//  Created by Ye Kuang on 3/27/17.
//  Copyright © 2017 Ye Kuang. All rights reserved.
//

#include "../include/shader_wrapper.h"

#include <iostream>

namespace pbf
{
    GLShaderWrapper::GLShaderWrapper(GLenum shader_type, const char* filepath)
    {
        Init(shader_type, filepath);
    }
    
    void GLShaderWrapper::Init(GLenum shader_type, const char* filepath)
    {
        shader_ = glCreateShader(shader_type);
        created_ = true;
        auto shader_source_str = ReadFile(filepath);
        const char* shader_source = shader_source_str.c_str();
        
        glShaderSource(shader_, 1, &shader_source, nullptr);
        glCompileShader(shader_);
        CHECK_SHADER_COMPILE_STATUS(shader_);
    }
    
    GLShaderWrapper::~GLShaderWrapper()
    {
        if (created_)
        {
            glDeleteShader(shader_);
            created_ = false;
        }
    }
} // namespace pbf
