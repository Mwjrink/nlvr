// use super::{buffer::*, texture::*};
// use ash::vk;
// use cgmath::Matrix4;

#[allow(dead_code)]
pub enum AlphaMode {
    OPAQUE,
    MASK,
    BLEND
}

// TODO this is being allowed for now because this is how the glTF spec specifies it, change this eventually
#[allow(non_snake_case)]
pub struct TextureInfo {
    pub index: u32, // Texture
    pub texCoord: u32,

    // pub extensions: // JSON object with extension-specific objects.
    // pub extras: // Application-specific data.
}

// TODO this is being allowed for now because this is how the glTF spec specifies it, change this eventually
#[allow(non_snake_case)]
pub struct PbrMetallicRoughness {
    pub baseColorFactor: [f32; 4],
    pub baseColorTexture: Option<TextureInfo>,
    pub metallicFactor: f32,
    pub roughnessFactor: f32,
    pub metallicRoughnessTexture: Option<TextureInfo>,
    // pub extensions: // JSON object with extension-specific objects.
    // pub extras: // Application-specific data.
}

impl Default for PbrMetallicRoughness {
    fn default() -> Self {
        Self {
            baseColorFactor: [1.0, 1.0, 1.0, 1.0],
            baseColorTexture: None,
            metallicFactor: 1.0,
            roughnessFactor: 1.0,
            metallicRoughnessTexture: None,
        }
    }
}

// TODO this is being allowed for now because this is how the glTF spec specifies it, change this eventually
#[allow(non_snake_case)]
pub struct NormalTextureInfo {
    pub index: u32,
    pub texCoord: u32,
    pub scale: f32,

    // pub extensions: // JSON object with extension-specific objects.
    // pub extras: // Application-specific data.
}

// TODO this is being allowed for now because this is how the glTF spec specifies it, change this eventually
#[allow(non_snake_case)]
pub struct OcclusionTextureInfo {
    pub index: u32,
    pub texCoord: u32,
    pub strength: f32,

    // pub extensions: // JSON object with extension-specific objects.
    // pub extras: // Application-specific data.
}

// TODO this is being allowed for now because this is how the glTF spec specifies it, change this eventually
#[allow(non_snake_case)]
pub struct Material {
    pub name: &'static str,
    // pub extensions: // JSON object with extension-specific objects.
    // pub extras: // Application-specific data.
    pub pbrMetallicRoughness: PbrMetallicRoughness,
    pub normalTexture: Option<NormalTextureInfo>,
    pub occlusionTexture: Option<OcclusionTextureInfo>,
    pub emissiveTexture: Option<TextureInfo>,
    pub emissiveFactor: [f32; 3],
    pub alphaMode: AlphaMode,
    pub alphaCutoff: f32,
    pub doubleSided: bool,
    // pub texture_index: Option<u32>,
    // pub texture: Option<Texture>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: Default::default(),
            pbrMetallicRoughness: Default::default(),
            normalTexture: None,
            occlusionTexture: None,
            emissiveTexture: None,
            emissiveFactor: [0.0, 0.0, 0.0],
            alphaMode: AlphaMode::OPAQUE,
            alphaCutoff: 0.5,
            doubleSided: false,
        }
    }
}

// {
//     "materials": [
//         {
//             "name": "Material0",
//             "pbrMetallicRoughness": {
//                 "baseColorFactor": [ 0.5, 0.5, 0.5, 1.0 ],
//                 "baseColorTexture": {
//                     "index": 1,
//                     "texCoord": 1
//                 },
//                 "metallicFactor": 1,
//                 "roughnessFactor": 1,
//                 "metallicRoughnessTexture": {
//                     "index": 2,
//                     "texCoord": 1
//                 }
//             },
//             "normalTexture": {
//                 "scale": 2,
//                 "index": 3,
//                 "texCoord": 1
//             },
//             "emissiveFactor": [ 0.2, 0.1, 0.0 ]
//         }
//     ]
// }
