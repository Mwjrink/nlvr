use super::buffer::*;
// use ash::vk;
// use cgmath::Matrix4;

#[allow(dead_code)]
struct Attributes {
    positions: u32,
    normals: u32,
    tangent: u32,
    texcoords: Vec<u32>,
}

pub struct Mesh {
    pub vertex_buffer_ptr: BuffPtr,
    pub vertex_count: usize,
    //
    pub index_buffer_ptr: BuffPtr,
    pub index_count: usize,
    // attributes: Attributes,
    // indices: u32,
    // material: u32,
    // mode: u32,
}

// {
//     "meshes": [
//         {
//             "primitives": [
//                 {
//                     "attributes": {
//                         "NORMAL": 23,
//                         "POSITION": 22,
//                         "TANGENT": 24,
//                         "TEXCOORD_0": 25
//                     },
//                     "indices": 21,
//                     "material": 3,
//                     "mode": 4
//                 }
//             ]
//         }
//     ]
// }
