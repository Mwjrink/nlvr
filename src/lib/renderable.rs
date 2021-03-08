use super::{buffer::*, texture::*};
// use ash::vk;
use cgmath::Matrix4;

pub struct Renderable {
    pub texture_index: u32,
    pub texture: Texture,
    //
    pub vertex_buffer_ptr: BuffPtr,
    pub vertex_count: usize,
    //
    pub index_buffer_ptr: BuffPtr,
    pub index_count: usize,
    //
    pub asset_path: String,
    //
    pub instances: Vec<Matrix4<f32>>,
}

impl Renderable {
    pub fn update(&mut self, instance_index: usize, transform: Matrix4<f32>) {
        self.instances[instance_index] = transform;
    }

    pub fn create_instance(&mut self, transform: Matrix4<f32>) -> usize {
        let index = self.instances.len();
        self.instances.push(transform);
        index
    }
}
