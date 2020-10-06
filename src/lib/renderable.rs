use super::{buffer::*, texture::*};
use ash::vk;
use cgmath::Matrix4;

pub struct Renderable {
    //
    model_index_count: usize,
    texture:           Texture,
    descriptor_pool:   vk::DescriptorPool,
    descriptor_sets:   Vec<vk::DescriptorSet,>,
    //
    uniform_buffers:   Vec<Buffer,>,
    vertex_buffer:     Buffer,
    index_buffer:      Buffer,
    //
    asset_path:        String,
}

impl Renderable {
    pub fn update(transform: &Matrix4<f32,>,) {
        // fjjf
    }
}
