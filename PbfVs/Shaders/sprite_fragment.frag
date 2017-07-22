#version 330 core

in vec3 vertex_color;
in vec3 vsPos3;
in float radius;

// out vec4 color;
out vec4 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

const float NEAR = 0.1; 
const float FAR  = 200.0; 

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * FAR * NEAR) / (FAR + NEAR - z * (FAR - NEAR));	
}

void main() {
    // compute normal
    vec3 normal;
    normal.xy = gl_PointCoord * 2 - 1.0;
    float r_sqr = dot(normal.xy, normal.xy);
    if (r_sqr > 1.0 - 1e-6) 
        discard;
    normal.z = -sqrt(1.0 - r_sqr);
    normal = normalize(normal);
    
    vec3 vsSphereSurfacePos = vsPos3 + normal * radius;
    float depthZ = LinearizeDepth(gl_FragCoord.z) / FAR;
    gl_FragDepth = depthZ;

    // defuse
    vec4 lightPos = (view * model * vec4(100.0, 100.0, -100.0, 1.0));
    vec3 lightDir = normalize(lightPos.xyz / lightPos.w - vsPos3);
    float diffuse = max(0.0f, dot(normal, lightDir));
     
    // color = vec4(abs(normal), 1.0f);
    // color = vec4(vec3(depthZ), 1.0f);
    color = vec4(vertex_color * diffuse, 1.0f);
}
