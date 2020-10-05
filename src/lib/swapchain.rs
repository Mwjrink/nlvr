use super::{context::*, image::*, queuefamilyindices::*, swapchainproperties::*, swapchainsupportdetails::*};
use ash::{extensions::khr::Swapchain, vk, Device};

pub struct SwapchainWrapper {
    swapchain:     Swapchain,
    swapchain_khr: vk::SwapchainKHR,
    properties:    SwapchainProperties,
    images:        Vec<Image,>,
}

impl SwapchainWrapper {
    /// Create the swapchain with optimal settings possible with
    /// `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    fn create_swapchain_and_images(
        vk_context: &VkContext,
        queue_families_indices: QueueFamiliesIndices,
        dimensions: [u32; 2],
    ) -> SwapchainWrapper
    {
        let details = SwapchainSupportDetails::new(
            vk_context.physical_device(),
            vk_context.surface(),
            vk_context.surface_khr(),
        );
        let properties = details.get_ideal_swapchain_properties(dimensions,);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: \
             {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let graphics = queue_families_indices.graphics_index;
        let present = queue_families_indices.present_index;
        let families_indices = [graphics, present,];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(vk_context.surface_khr(),)
                .min_image_count(image_count,)
                .image_format(format.format,)
                .image_color_space(format.color_space,)
                .image_extent(extent,)
                .image_array_layers(1,)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT,);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT,)
                    .queue_family_indices(&families_indices,)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE,)
            };

            builder
                .pre_transform(details.capabilities.current_transform,)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE,)
                .present_mode(present_mode,)
                .clipped(true,)
                .build()
            // .old_swapchain() We don't have an old swapchain but can't pass null
        };

        let swapchain = Swapchain::new(vk_context.instance(), vk_context.device(),);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None,).unwrap() };
        let vkImages = unsafe { swapchain.get_swapchain_images(swapchain_khr,).unwrap() };

        let images = vkImages
            .iter()
            .map(|vkImage| Image {
                image:  *vkImage,
                memory: vk::DeviceMemory::null(),
                view:   None,
            },)
            .collect();

        Self {
            swapchain,
            swapchain_khr,
            properties,
            images,
        }
    }

    /// Create one image view for each image of the swapchain.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[Image],
        swapchain_properties: SwapchainProperties,
    )
    {
        swapchain_images
            .iter()
            .map(|image| {
                image.create_image_view(
                    device,
                    1,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            },)
            .collect::<Vec<_,>>();
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    pub fn recreate_swapchain(&mut self,) {
        log::debug!("Recreating swapchain.");

        self.wait_gpu_idle();

        self.cleanup_swapchain();

        let device = self.vk_context.device();

        let dimensions = [
            self.swapchain_properties.extent.width,
            self.swapchain_properties.extent.height,
        ];
        let (swapchain, swapchain_khr, properties, images,) =
            Self::create_swapchain_and_images(&self.vk_context, self.queue_families_indices, dimensions,);
        let swapchain_image_views = Self::create_swapchain_image_views(device, &images, properties,);

        let render_pass = Self::create_render_pass(device, properties, self.msaa_samples, self.depth_format,);
        let (pipeline, layout,) = Self::create_pipeline(
            device,
            properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
        );

        let color_texture = Self::create_color_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            properties,
            self.msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            self.depth_format,
            properties.extent,
            self.msaa_samples,
        );

        let swapchain_framebuffers = Self::create_framebuffers(
            device,
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            self.vertex_buffer,
            self.index_buffer,
            self.model_index_count,
            layout,
            &self.descriptor_sets,
            pipeline,
        );

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_layout = layout;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.command_buffers = command_buffers;
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup_swapchain(&mut self,) {
        let device = self.vk_context.device();
        unsafe {
            self.depth_texture.destroy(device,);
            self.color_texture.destroy(device,);
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None,),);
            device.free_command_buffers(self.command_pool, &self.command_buffers,);
            device.destroy_pipeline(self.pipeline, None,);
            device.destroy_pipeline_layout(self.pipeline_layout, None,);
            device.destroy_render_pass(self.render_pass, None,);
            self.swapchain_image_views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None,),);
            self.swapchain.destroy_swapchain(self.swapchain_khr, None,);
        }
    }
}
