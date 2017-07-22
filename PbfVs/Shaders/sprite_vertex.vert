// sprite vertex
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 vertex_color;
out vec3 vsPos3;
out float radius;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
    // view space (eye space) position
    vec4 vsPosition = view * model * vec4(position.xyz, 1.0);
    // clip space (NDC) position, all coordinate components are in [-1.0, 1.0]
    gl_Position = proj * vsPosition;

    vsPos3 = vsPosition.xyz / vsPosition.w;
    float vsPositionLen = length(vsPos3);
    radius = max(3.0, 600.0 / vsPositionLen);
    // radius = 36000.0 / vsPositionLen / vsPositionLen;
    // Need to glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    gl_PointSize = radius;
      
    vertex_color = color;
}
