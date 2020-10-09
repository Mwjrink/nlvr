use ash::vk;
use std::mem::size_of;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Vertex {
    pub pos:    [f32; 3],
    // color:  [f32; 3],
    pub coords: [f32; 2],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0,)
            .stride(size_of::<Vertex,>() as _,)
            .input_rate(vk::VertexInputRate::VERTEX,)
            .build()
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let position_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0,)
            .location(0,)
            .format(vk::Format::R32G32B32_SFLOAT,)
            .offset(0,)
            .build();
        // let color_desc = vk::VertexInputAttributeDescription::builder()
        //     .binding(0,)
        //     .location(1,)
        //     .format(vk::Format::R32G32B32_SFLOAT,)
        //     .offset(12,)
        //     .build();
        let coords_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0,)
            .location(1,)
            .format(vk::Format::R32G32_SFLOAT,)
            .offset(12,)
            .build();
        // color_desc,
        [position_desc, coords_desc,]
    }
}
