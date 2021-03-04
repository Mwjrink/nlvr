use ash::vk;
use cgmath::Matrix4;

pub trait UBO {
    fn get_descriptor_set_layout_binding() -> vk::DescriptorSetLayoutBinding;
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct CameraUBO {
    pub view: Matrix4<f32,>,
    pub proj: Matrix4<f32,>,
}

impl UBO for CameraUBO {
    fn get_descriptor_set_layout_binding() -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            // .immutable_samplers() null since we're not creating a sampler descriptor
            .build()
    }
}
