#include "../include/shader_program.h"
#include "../include/shader_wrapper.h"

namespace pbf {
void ShaderProgram::Init(const char *vert_filepath, const char *frag_filepath) {
  // Shaders only need to be created once. Once they are
  // attached to a shader program, they can be deleted safely.
  GLShaderWrapper vert_shader(GL_VERTEX_SHADER, vert_filepath);
  GLShaderWrapper frag_shader(GL_FRAGMENT_SHADER, frag_filepath);

  shader_program_ = glCreateProgram();
  glAttachShader(shader_program_, vert_shader.Get());
  glAttachShader(shader_program_, frag_shader.Get());
  glLinkProgram(shader_program_);

  CHECK_PROGRAM_LINK_STATUS(shader_program_);
}

void ShaderProgram::Use() const { glUseProgram(shader_program_); }

void ShaderProgram::Unbind() const { glUseProgram(0); }

GLuint ShaderProgram::GetUniformLoc(const char *name) const {
  return glGetUniformLocation(Get(), name);
}

WithShaderProgram::WithShaderProgram(const ShaderProgram &p) : p_(p) {
  p_.Use();
}
WithShaderProgram::~WithShaderProgram() { p_.Unbind(); }
} // namespace pbf
