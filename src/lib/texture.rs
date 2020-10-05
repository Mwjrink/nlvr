use super::{buffer::*, context::*, fs, image::*, vulkan::*};
use ash::{version::{DeviceV1_0, InstanceV1_0},
          vk,
          Device};
use std::mem::{align_of, size_of};

#[derive(Clone, Copy)]
pub struct Texture {
    pub image:   Image,
    pub sampler: Option<vk::Sampler,>,
}

impl Texture {
    pub fn destroy(&mut self, device: &Device,) {
        unsafe {
            if let Some(sampler,) = self.sampler.take() {
                device.destroy_sampler(sampler, None,);
            }
            self.image.destroy(device,);
        }
    }

    pub fn create_texture_image(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        copy_queue: vk::Queue,
        file_name: &str,
    ) -> Texture
    {
        let cursor = fs::load(file_name,);
        let image = image::load(cursor, image::ImageFormat::Jpeg,).unwrap().flipv();
        let image_as_rgb = image.to_rgba();
        let width = (&image_as_rgb).width();
        let height = (&image_as_rgb).height();
        let max_mip_levels = ((width.min(height,) as f32).log2().floor() + 1.0) as u32;
        let extent = vk::Extent2D {
            width,
            height,
        };
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8,>()) as vk::DeviceSize;
        let device = vk_context.device();

        let buffer = Buffer::create_buffer(
            vk_context,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = device
                .map_memory(buffer.memory, 0, image_size, vk::MemoryMapFlags::empty(),)
                .unwrap();
            let mut align = ash::util::Align::new(ptr, align_of::<u8,>() as _, buffer.size,);
            align.copy_from_slice(&pixels,);
            device.unmap_memory(buffer.memory,);
        }

        let image = Image::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            image.transition_image_layout(
                device,
                command_pool,
                copy_queue,
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            Image::copy_from_buffer(device, command_pool, copy_queue, buffer, image, extent,);
            // Self::copy_buffer_to_image(device, command_pool, copy_queue, bufferMagic, image, extent,);

            Self::generate_mipmaps(
                vk_context,
                command_pool,
                copy_queue,
                image,
                extent,
                vk::Format::R8G8B8A8_UNORM,
                max_mip_levels,
            );
        }

        unsafe {
            device.destroy_buffer(buffer.buffer, None,);
            device.free_memory(image.memory, None,);
        }

        image.create_image_view(
            device,
            max_mip_levels,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        );

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR,)
                .min_filter(vk::Filter::LINEAR,)
                .address_mode_u(vk::SamplerAddressMode::REPEAT,)
                .address_mode_v(vk::SamplerAddressMode::REPEAT,)
                .address_mode_w(vk::SamplerAddressMode::REPEAT,)
                .anisotropy_enable(true,)
                .max_anisotropy(16.0,)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK,)
                .unnormalized_coordinates(false,)
                .compare_enable(false,)
                .compare_op(vk::CompareOp::ALWAYS,)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR,)
                .mip_lod_bias(0.0,)
                .min_lod(0.0,)
                .max_lod(max_mip_levels as _,)
                .build();

            unsafe { device.create_sampler(&sampler_info, None,).unwrap() }
        };

        Self {
            image,
            sampler: Some(sampler,),
        }
    }

    fn generate_mipmaps(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        image: Image,
        extent: vk::Extent2D,
        format: vk::Format,
        mip_levels: u32,
    )
    {
        let format_properties = unsafe {
            vk_context
                .instance()
                .get_physical_device_format_properties(vk_context.physical_device(), format,)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,)
        {
            panic!("Linear blitting is not supported for format {:?}.", format)
        }

        execute_one_time_commands(vk_context.device(), command_pool, transfer_queue, |buffer| {
            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(image.image,)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED,)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED,)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    layer_count: 1,
                    level_count: 1,
                    ..Default::default()
                },)
                .build();

            let mut mip_width = extent.width as i32;
            let mut mip_height = extent.height as i32;
            for level in 1..mip_levels {
                let next_mip_width = if mip_width > 1 { mip_width / 2 } else { mip_width };
                let next_mip_height = if mip_height > 1 { mip_height / 2 } else { mip_height };

                barrier.subresource_range.base_mip_level = level - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
                let barriers = [barrier,];

                unsafe {
                    vk_context.device().cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
                };

                let blit = vk::ImageBlit::builder()
                    .src_offsets([
                        vk::Offset3D {
                            x: 0, y: 0, z: 0,
                        },
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ],)
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        mip_level:        level - 1,
                        base_array_layer: 0,
                        layer_count:      1,
                    },)
                    .dst_offsets([
                        vk::Offset3D {
                            x: 0, y: 0, z: 0,
                        },
                        vk::Offset3D {
                            x: next_mip_width,
                            y: next_mip_height,
                            z: 1,
                        },
                    ],)
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask:      vk::ImageAspectFlags::COLOR,
                        mip_level:        level,
                        base_array_layer: 0,
                        layer_count:      1,
                    },)
                    .build();
                let blits = [blit,];

                unsafe {
                    vk_context.device().cmd_blit_image(
                        buffer,
                        image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &blits,
                        vk::Filter::LINEAR,
                    )
                };

                barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                let barriers = [barrier,];

                unsafe {
                    vk_context.device().cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
                };

                mip_width = next_mip_width;
                mip_height = next_mip_height;
            }

            barrier.subresource_range.base_mip_level = mip_levels - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
            let barriers = [barrier,];

            unsafe {
                vk_context.device().cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                )
            };
        },);
    }
}
