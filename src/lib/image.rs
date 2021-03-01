use super::{buffer::*, context::*, utils::*};//, vulkan::*};
use ash::{version::DeviceV1_0, vk, Device};

#[derive(Clone, Copy)]
pub struct Image {
    pub image:  vk::Image,
    pub memory: vk::DeviceMemory,
    pub view:   Option<vk::ImageView,>,
}

impl Image {
    pub unsafe fn destroy(&mut self, device: &Device,) {
        if let Some(view,) = self.view {
            device.destroy_image_view(view, None,);
        }

        device.destroy_image(self.image, None,);
        device.free_memory(self.memory, None,);
    }

    pub fn create_image(
        vk_context: &VkContext,
        mem_properties: vk::MemoryPropertyFlags,
        extent: vk::Extent2D,
        mip_levels: u32,
        sample_count: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> Image
    {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D,)
            .extent(vk::Extent3D {
                width:  extent.width,
                height: extent.height,
                depth:  1,
            },)
            .mip_levels(mip_levels,)
            .array_layers(1,)
            .format(format,)
            .tiling(tiling,)
            .initial_layout(vk::ImageLayout::UNDEFINED,)
            .usage(usage,)
            .sharing_mode(vk::SharingMode::EXCLUSIVE,)
            .samples(sample_count,)
            .flags(vk::ImageCreateFlags::empty(),)
            .build();

        let device = vk_context.device();
        let image = unsafe { device.create_image(&image_info, None,).unwrap() };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image,) };
        let mem_type_index = Buffer::find_memory_type(mem_requirements, vk_context.get_mem_properties(), mem_properties,);

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size,)
            .memory_type_index(mem_type_index,)
            .build();
        let memory = unsafe {
            let mem = device.allocate_memory(&alloc_info, None,).unwrap();
            device.bind_image_memory(image, mem, 0,).unwrap();
            mem
        };

        Image {
            image,
            memory,
            view: None,
        }
    }

    pub fn create_image_view(
        &mut self,
        device: &Device,
        mip_levels: u32,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    )
    {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(self.image,)
            .view_type(vk::ImageViewType::TYPE_2D,)
            .format(format,)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },)
            .build();

        self.view = Some(unsafe { device.create_image_view(&create_info, None,).unwrap() },);
    }

    pub fn transition_image_layout(
        &mut self,
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        // image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    )
    {
        execute_one_time_commands(device, command_pool, transition_queue, |buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage,) = match (old_layout, new_layout,) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ),
                _ => panic!("Unsupported layout transition({:?} => {:?}).", old_layout, new_layout),
            };

            let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = vk::ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format,) {
                    mask |= vk::ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(old_layout,)
                .new_layout(new_layout,)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED,)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED,)
                .image(self.image,)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                },)
                .src_access_mask(src_access_mask,)
                .dst_access_mask(dst_access_mask,)
                .build();
            let barriers = [barrier,];

            unsafe {
                device.cmd_pipeline_barrier(
                    buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                )
            };
        },);
    }

    pub fn has_stencil_component(format: vk::Format,) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    pub fn copy_from_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        buffer: &Buffer,
        image: Image,
        extent: vk::Extent2D,
    )
    {
        execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0,)
                .buffer_row_length(0,)
                .buffer_image_height(0,)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask:      vk::ImageAspectFlags::COLOR,
                    mip_level:        0,
                    base_array_layer: 0,
                    layer_count:      1,
                },)
                .image_offset(vk::Offset3D {
                    x: 0, y: 0, z: 0,
                },)
                .image_extent(vk::Extent3D {
                    width:  extent.width,
                    height: extent.height,
                    depth:  1,
                },)
                .build();
            let regions = [region,];
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer.buffer,
                    image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        },)
    }

    // Generalize this
    // pub fn create_color_texture(
    //     vk_context: &VkContext,
    //     command_pool: vk::CommandPool,
    //     transition_queue: vk::Queue,
    //     swapchain_properties: SwapchainProperties,
    //     msaa_samples: vk::SampleCountFlags,
    // ) -> Texture
    // {
    //     // TODO make this a function in image
    //     let format = swapchain_properties.format.format;
    //     let (image, memory,) = Self::create_image(
    //         vk_context,
    //         vk::MemoryPropertyFlags::DEVICE_LOCAL,
    //         swapchain_properties.extent,
    //         1,
    //         msaa_samples,
    //         format,
    //         vk::ImageTiling::OPTIMAL,
    //         vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
    //     );

    //     Self::transition_image_layout(
    //         vk_context.device(),
    //         command_pool,
    //         transition_queue,
    //         image,
    //         1,
    //         format,
    //         vk::ImageLayout::UNDEFINED,
    //         vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    //     );

    //     let view = Self::create_image_view(vk_context.device(), image, 1, format, vk::ImageAspectFlags::COLOR,);

    //     Texture::new(image, memory, view, None,)
    // }
}
