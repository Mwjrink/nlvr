use super::{buffer::*, texture::*};
// use ash::vk;
use cgmath::Matrix4;

pub struct Renderable {
    // TODO textures dont exist for now
    // pub texture:           Texture,

    pub vertex_buffer_ptr: BuffPtr,
    pub index_buffer_ptr:  BuffPtr,
    pub model_index_count: usize,
    
    // pub uniform_buffers:   Vec<Buffer,>,
    
    pub asset_path:        String,
    
    pub instances:         Vec<Matrix4<f32,>,>,
}

impl Renderable {
    pub fn update(&mut self, instance_index: usize, transform: Matrix4<f32,>,) {
        self.instances[instance_index] = transform;
    }

    pub fn create_instance(&mut self, transform: Matrix4<f32,>,) -> usize {
        let index = self.instances.len();
        self.instances.push(transform,);
        index
    }

    pub fn rebuild() {
        // let command_buffers = Self::create_and_register_command_buffers(
        //     device,
        //     self.command_pool,
        //     &self.framebuffers,
        //     render_pass,
        //     swapchainWrapper.properties,
        //     self.vertex_buffer,
        //     self.index_buffer,
        //     self.model_index_count,
        //     layout,
        //     &self.descriptor_sets,
        //     pipeline,
        // );
    }
}
