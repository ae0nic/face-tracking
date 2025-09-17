#version 330

uniform sampler2D p3d_Texture1;

// Input from vertex shader
in vec2 texcoord;
in vec3 NORMAL;
in vec3 POSITION;

uniform mat4 p3d_ViewMatrix;
uniform mat4 p3d_ViewMatrixInverse;

const float PI = 3.1415926535897932384626433832795;

uniform int LIGHTS;
uniform int DEBUG_MODE;

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
    vec3 VIEW = vec3(p3d_ViewMatrix[2][0], p3d_ViewMatrix[2][1], p3d_ViewMatrix[2][2]);

    vec3 DIFFUSE_LIGHT = vec3(0.);
    vec3 AMBIENT_LIGHT = vec3(0.6);
    vec3 SPECULAR_LIGHT = vec3(0);

    float roughness = .8;
    float power = 5;
    float multi = 0.9;


    vec3 d = vec3(0);

    float[] allContribtions = float[20](0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);

    float DOT = 0.;
    for (int i = 0; i < LIGHTS; i++) {
        if (p3d_LightSource[i].color == vec4(0)) {
            break;
        }
        bool IS_DIRECTIONAL = p3d_LightSource[i].position.w == 0.;
        vec3 direction = vec3(0);
        float directionalMultiplier = 1.;
        if (IS_DIRECTIONAL) {
            direction = -(p3d_ViewMatrixInverse * p3d_LightSource[i].position).xyz;
            directionalMultiplier = 1.2;
            d = -direction;
        } else {
            direction = normalize(POSITION - (p3d_ViewMatrixInverse * p3d_LightSource[i].position).xyz);
        }
        vec3 LIGHT_COLOR = p3d_LightSource[i].color.xyz;
        float ATTENUATION = p3d_LightSource[i].constantAttenuation;

        vec3 contribution = round(clamp(dot(NORMAL, direction), 0.0, 1.0) * directionalMultiplier * ATTENUATION * LIGHT_COLOR);
        contribution *= multi;
        contribution = pow(contribution, vec3(power));
        contribution /= multi;
        contribution = clamp(contribution, 0., 1.);

        allContribtions[i] = length(contribution);

        DIFFUSE_LIGHT += contribution;

        vec3 R = -reflect(direction, NORMAL);
        float RdotV = dot(R, VIEW);
        float mid = 1.0 - roughness;
        mid *= mid;
        float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid * 0.5;
        SPECULAR_LIGHT += 1.0 * clamp(round(intensity + 0.5) - 0.5 - roughness / 2., 0, 1) * ATTENUATION * LIGHT_COLOR;
    }
    DIFFUSE_LIGHT = clamp(round(DIFFUSE_LIGHT), 0, 0.8);

    vec3 AMB_OR_DIFFUSE = AMBIENT_LIGHT;
    if (length(DIFFUSE_LIGHT) > 0) {
      AMB_OR_DIFFUSE = DIFFUSE_LIGHT;
    }

    vec4 color = p3d_Material.emission + vec4(texture(p3d_Texture1, texcoord).rgb * (SPECULAR_LIGHT + AMB_OR_DIFFUSE), texture(p3d_Texture1, texcoord).a);
    if (DEBUG_MODE == 1) {
        color = vec4(vec3(allContribtions[0]), 1.);
    } else if (DEBUG_MODE == 2) {
        color = vec4(vec3(allContribtions[1]), 1.);
    } else if (DEBUG_MODE == 9) {
        color = vec4(NORMAL, 1.0);
    }

    p3d_FragColor = color;
}