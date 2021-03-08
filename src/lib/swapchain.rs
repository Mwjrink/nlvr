use super::{context::*,
            image::*,
            queuefamilyindices::*,
            swapchainproperties::*,
            swapchainsupportdetails::*,
            texture::*};
use ash::{extensions::khr::Swapchain, version::DeviceV1_0, vk, Device};

pub struct SwapchainWrapper {
    pub swapchain:     Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub properties:    SwapchainProperties,
    pub images:        Vec<Image,>,
    pub color_texture: Texture,
    pub depth_texture: Texture,
    /* VkInstance instance;
     * VkDevice device;
     * VkPhysicalDevice physicalDevice;
     * VkSurfaceKHR surface;
     * VkFormat colorFormat;
     * VkColorSpaceKHR colorSpace;
     * VkSwapchainKHR swapChain = VK_NULL_HANDLE;
     * uint32_t imageCount;
     * std::vector<VkImage> images;
     * std::vector<SwapChainBuffer> buffers;
     * uint32_t queueNodeIndex = UINT32_MAX;
     */
}

impl SwapchainWrapper {
    /// Create the swapchain with optimal settings possible with
    /// `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    pub fn create_swapchain_and_images(
        vk_context: &VkContext,
        queue_families_indices: QueueFamiliesIndices,
        dimensions: [u32; 2],
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
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
        let vk_images = unsafe { swapchain.get_swapchain_images(swapchain_khr,).unwrap() };

        let mut images = vk_images
            .iter()
            .map(|vk_image| Image {
                image:  *vk_image,
                memory: vk::DeviceMemory::null(),
                view:   None,
            },)
            .collect::<Vec<Image,>>();

        let device = vk_context.device();

        for image in &mut images {
            image.create_image_view(device, 1, properties.format.format, vk::ImageAspectFlags::COLOR,);
        }

        let color_texture =
            Texture::create_color_texture(&vk_context, command_pool, transition_queue, properties, msaa_samples,);

        let depth_texture = Texture::create_depth_texture(
            &vk_context,
            command_pool,
            transition_queue,
            depth_format,
            properties.extent,
            msaa_samples,
        );

        Self {
            swapchain,
            swapchain_khr,
            properties,
            images,
            color_texture,
            depth_texture,
        }
    }

    /// Clean up the swapchain and all resources that depend on it.
    pub fn cleanup(&mut self, device: &Device,) {
        unsafe {
            self.depth_texture.destroy(device,);
            self.color_texture.destroy(device,);
            self.images
                .iter()
                .for_each(|v| device.destroy_image_view(v.view.unwrap(), None,),);
            self.swapchain.destroy_swapchain(self.swapchain_khr, None,);
        }
    }
}
