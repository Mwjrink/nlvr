#version 460
#extension GL_EXT_nonuniform_qualifier: enable

layout(push_constant) uniform Index
{
    layout(offset = 64) uint index;
} index;

layout(binding = 1) uniform sampler2D texSampler[];

layout(location = 0) in vec2 fragCoords;

layout(location = 0) out vec4 outColor;

void main(){
    outColor = texture(texSampler[index.index], fragCoords);
}
