use ash::{extensions::{ext::DebugReport,
                       khr::{Surface, Swapchain}},
          version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
          vk,
          Device,
          Entry,
          Instance};

pub struct Renderable {
    // texture
    // vertex_buffer
    // vertex_buffer_memory
    // index_buffer
    // index_buffer_memory
    texture:              Texture,
    vertex_buffer:        vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer:         vk::Buffer,
    index_buffer_memory:  vk::DeviceMemory,
}

impl Component for Renderable {
    //
}
