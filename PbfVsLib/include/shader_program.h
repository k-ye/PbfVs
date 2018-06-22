#ifndef shader_program_h
#define shader_program_h

#include "gl_utils.h"

namespace pbf {
// A wrapper class that initializes a GL program using both vertex and fragment
// shaders
class ShaderProgram {
public:
  ShaderProgram() = default;

  void Init(const char *vert_filepath, const char *frag_filepath);

  inline GLuint Get() const { return shader_program_; }

  void Use() const;

  void Unbind() const;

  GLuint GetUniformLoc(const char *name) const;

private:
  GLuint shader_program_{0};
};

// RAII
class WithShaderProgram {
public:
  WithShaderProgram(const ShaderProgram &p);
  ~WithShaderProgram();

private:
  const ShaderProgram &p_;
};

} // namespace pbf

#endif // shader_program_h
