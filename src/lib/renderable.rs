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
    vertex_buffer_ptr: buff_ptr,
    index_buffer_ptr:  buff_ptr,
    //
    asset_path:        String,
    // command_buffers:   Vec<vk::CommandBuffer,>,
}

impl Renderable {
    pub fn update(transform: &Matrix4<f32,>,) {
        // fjjf
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
