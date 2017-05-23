#version 330 core

in vec3 vertex_color;
out vec4 color;

void main()
{
    // color = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    color = vec4(vertex_color, 1.0f);
}
