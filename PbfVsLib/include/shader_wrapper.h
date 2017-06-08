#ifndef shader_wrapper_h
#define shader_wrapper_h

#include "gl_utils.h"

#include <string>

namespace pbf
{

class GLShaderWrapper
{
public:
    GLShaderWrapper() = default;
    GLShaderWrapper(GLenum shader_type, const char* filepath);
    ~GLShaderWrapper();
    
    void Init(GLenum shader_type, const char* filepath);
    inline GLuint Get() const { return shader_; }
    
private:
    GLuint shader_;
    bool created_{false};
};
    
} // namespace pbf

#endif /* shader_wrapper_h */
