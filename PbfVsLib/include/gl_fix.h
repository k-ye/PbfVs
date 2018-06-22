#ifndef gl_fix_h
#define gl_fix_h

// Compatible issue
// http://stackoverflow.com/questions/22364937/how-to-use-opengl-3-0-on-macos-with-intel-hd-3000
#define GLFW_FORWARD_COMPATIBLE()                                              \
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

// Flashing like hell issue
// http://stackoverflow.com/questions/27678819/crazy-flashing-window-opengl-glfw
#define GL_STABLE_FRAME() glClear(GL_COLOR_BUFFER_BIT)

#endif /* gl_fix_h */
