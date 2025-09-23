#version 330

uniform sampler2D p3d_Texture0;

// Input from vertex shader
in vec2 texcoord;
in vec3 NORMAL;
in vec3 POSITION;

uniform mat4 p3d_ViewMatrix;
uniform mat4 p3d_ViewMatrixInverse;

const float PI = 3.1415926535897932384626433832795;

uniform int LIGHTS;

uniform struct p3d_LightSourceParameters {
    vec4 color;
    vec4 position;
    float constantAttenuation;
} p3d_LightSource[20];

uniform struct p3d_MaterialParameters {
  vec4 baseColor;
  vec4 emission;
} p3d_Material;

// Output to the screen
out vec4 p3d_FragColor;



void main() {
    vec3 hair_color =  1.2 * normalize(vec3(0.2, 0.3, 0.5));

    vec4 color = p3d_Material.emission + vec4((0.6+0.4*texture(p3d_Texture0, texcoord).rgb) * hair_color, 1.);

    p3d_FragColor = color;
}