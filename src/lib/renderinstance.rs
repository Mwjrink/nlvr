use super::{buffer::*,
            camera::*,
            context::*,
            debug::*,
            fs,
            image::*,
            inflightframedata::*,
            math,
            queuefamilyindices::*,
            renderable::*,
            shaders::*,
            swapchain::*,
            swapchainproperties::*,
            swapchainsupportdetails::*,
            texture::*,
            ubo::*,
            utils::*,
            vertex::*,
            vulkan::*};
use ash::{extensions::{ext::DebugReport,
                       khr::{Surface, Swapchain}},
          version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
          vk,
          Device,
          Entry,
          Instance};
use cgmath::{vec3, Deg, Matrix4, Point3, Vector3};
use std::{ffi::{CStr, CString},
          mem::{align_of, size_of}};
use winit::window::Window;

pub struct RenderInstance<'a,> {
    vk_context:             VkContext,
    //
    pipeline_layout:        vk::PipelineLayout,
    pipeline:               vk::Pipeline,
    command_buffers:        Vec<vk::CommandBuffer,>,
    msaa_samples:           vk::SampleCountFlags,
    descriptor_set_layout:  vk::DescriptorSetLayout,
    swapchain:              SwapchainWrapper,
    render_pass:            vk::RenderPass,
    //
    command_pool:           vk::CommandPool,
    transient_command_pool: vk::CommandPool,
    in_flight_frames:       InFlightFrames,
    //
    framebuffers:           Vec<vk::Framebuffer,>,
    vertex_binding_descs:   vk::VertexInputBindingDescription,
    vertex_attribute_descs: &'a [vk::VertexInputAttributeDescription],
    model_index_count:      usize,
    texture:                Texture,
    descriptor_pool:        vk::DescriptorPool,
    descriptor_sets:        Vec<vk::DescriptorSet,>,
    //
    graphics_queue:         vk::Queue,
    present_queue:          vk::Queue,
    // make these two Option<vk::Queue> and then use them if they exist but fall back on graphics queue
    transfer_queue:         vk::Queue,
    // this is the master pool, the command buffers are run on this and they consist of many models, vertices, indices
    // and texture index for a texture array
    vertex_buffer:          vk::Buffer,
    vertex_buffer_memory:   vk::DeviceMemory,
    index_buffer:           vk::Buffer,
    index_buffer_memory:    vk::DeviceMemory,
    //
    frames_in_flight:       usize,
    //
    depth_format:           vk::Format,
}

impl RenderInstance<'_,> {
    pub fn create(window: &Window,) -> RenderInstance {
        log::debug!("Creating application.");

        let entry = Entry::new().expect("Failed to create entry.",);
        let instance = Self::create_instance(&entry, window,);

        let surface = Surface::new(&entry, &instance,);
        let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, window, None,).unwrap() };

        let debug_report_callback = setup_debug_messenger(&entry, &instance,);

        let (physical_device, queue_families_indices,) = Self::pick_physical_device(&instance, &surface, surface_khr,);

        // move to a queue class potentially
        let (device, graphics_queue, present_queue,) =
            Self::create_logical_device_with_graphics_queue(&instance, physical_device, queue_families_indices,);

        let vk_context = VkContext::new(
            entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            queue_families_indices,
        );

        let (swapchain, swapchain_khr, properties, images,) =
            Self::create_swapchain_and_images(&vk_context, queue_families_indices, [
                window.inner_size().width,
                window.inner_size().height,
            ],);
        let swapchain_image_views = Self::create_swapchain_image_views(vk_context.device(), &images, properties,);

        let msaa_samples = vk_context.get_max_usable_sample_count();
        let depth_format = Self::find_depth_format(&vk_context,);

        // Breakpoint

        let render_pass = Self::create_render_pass(vk_context.device(), properties, msaa_samples, depth_format,);
        let descriptor_set_layout =
            Self::create_descriptor_set_layout(vk_context.device(), CameraUBO::get_descriptor_set_layout_binding(),);
        let (pipeline, layout,) = Self::create_pipeline(
            vk_context.device(),
            properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
            Vertex::get_binding_description(),
            &Vertex::get_attribute_descriptions(),
        );

        log::debug!("Create command pool");

        let command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::empty(),
        );

        log::debug!("Create transient command pool");

        let transient_command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        log::debug!("Create color texture");

        let color_texture =
            Self::create_color_texture(&vk_context, command_pool, graphics_queue, properties, msaa_samples,);

        log::debug!("Create depth texture");

        let depth_texture = Self::create_depth_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            depth_format,
            properties.extent,
            msaa_samples,
        );

        log::debug!("Create swapchain framebuffers");

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        log::debug!("Create texture image");

        let texture = Self::create_texture_image(&vk_context, command_pool, graphics_queue,);

        log::debug!("Load model");

        let (vertices, indices,) = Self::load_model();
        let (vertex_buffer, vertex_buffer_memory,) =
            Self::create_vertex_buffer(&vk_context, transient_command_pool, graphics_queue, &vertices,);
        let (index_buffer, index_buffer_memory,) =
            Self::create_index_buffer(&vk_context, transient_command_pool, graphics_queue, &indices,);
        let (uniform_buffers, uniform_buffer_memories,) = Self::create_uniform_buffers(&vk_context, images.len(),);

        log::debug!("Create descriptor sets & pool");

        let descriptor_pool = Self::create_descriptor_pool(vk_context.device(), images.len() as _,);
        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            texture,
        );

        log::debug!("Create and register command buffers");

        let command_buffers = Self::create_and_register_command_buffers(
            vk_context.device(),
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            vertex_buffer,
            index_buffer,
            indices.len(),
            layout,
            &descriptor_sets,
            pipeline,
        );

        log::debug!("Create sync objects");

        let in_flight_frames = Self::create_sync_objects(vk_context.device(),);

        Self {
            camera: Default::default(),
            is_left_clicked: false,
            cursor_position: [0, 0,],
            cursor_delta: None,
            wheel_delta: None,
            //
            vk_context,
            queue_families_indices,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            swapchain_image_views,
            render_pass,
            descriptor_set_layout,
            pipeline_layout: layout,
            pipeline,
            swapchain_framebuffers,
            command_pool,
            transient_command_pool,
            msaa_samples,
            color_texture,
            depth_format,
            depth_texture,
            texture,
            model_index_count: indices.len(),
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffer_memories,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            in_flight_frames,
        }

        // vk_context:             VkContext,
        // pipeline_layout:        vk::PipelineLayout,
        // pipeline:               vk::Pipeline,
        // command_buffers:        Vec<vk::CommandBuffer,>,
        // msaa_samples:           vk::SampleCountFlags,
        // descriptor_set_layout:  vk::DescriptorSetLayout,
        // swapchain:              SwapchainWrapper,
        // render_pass:            vk::RenderPass,
        // command_pool:           vk::CommandPool,
        // transient_command_pool: vk::CommandPool,
        // in_flight_frames:       InFlightFrames,
        // framebuffers:           Vec<vk::Framebuffer,>,
        // vertex_binding_descs:   vk::VertexInputBindingDescription,
        // vertex_attribute_descs: &'a [vk::VertexInputAttributeDescription],
        // model_index_count:      usize,
        // texture:                Texture,
        // descriptor_pool:        vk::DescriptorPool,
        // descriptor_sets:        Vec<vk::DescriptorSet,>,
        // graphics_queue:         vk::Queue,
        // present_queue:          vk::Queue,
        // transfer_queue:         vk::Queue,
        // vertex_buffer:          vk::Buffer,
        // vertex_buffer_memory:   vk::DeviceMemory,
        // index_buffer:           vk::Buffer,
        // index_buffer_memory:    vk::DeviceMemory,
        // frames_in_flight:       usize,
        // depth_format:           vk::Format,
    }

    pub fn create_renderable_from_file(&self, asset_path: String,) -> Renderable {
        // TODO: use reader

        asset_path.push_str(".jpg",);
        let texture = Texture::create_texture_image(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            asset_path.as_str(),
        );

        let (vertices, indices,) = Self::load_model();
        let vertex_buffer = Self::create_vertex_buffer(
            &self.vk_context,
            self.transient_command_pool,
            self.graphics_queue,
            &vertices,
        );
        let index_buffer = Self::create_index_buffer(
            &self.vk_context,
            self.transient_command_pool,
            self.graphics_queue,
            &indices,
        );
        let uniform_buffers = Self::create_uniform_buffers(&self.vk_context, self.frames_in_flight,);

        let descriptor_pool = Self::create_descriptor_pool(self.vk_context.device(), self.frames_in_flight as _,);
        let descriptor_sets = Self::create_descriptor_sets(
            self.vk_context.device(),
            descriptor_pool,
            self.descriptor_set_layout,
            &uniform_buffers
                .iter()
                .map(|buff| buff.buffer,)
                .collect::<Vec<vk::Buffer,>>(),
            texture,
        );

        Renderable {
            model_index_count: indices.len(),
            texture,
            descriptor_pool,
            descriptor_sets,
            //
            uniform_buffers,
            vertex_buffer,
            index_buffer,
            //
            asset_path,
        }
    }

    fn load_model() -> (Vec<Vertex,>, Vec<u32,>,) {
        log::debug!("Loading model.");
        let mut cursor = fs::load("models/chalet.obj",);
        let (models, _,) =
            tobj::load_obj_buf(&mut cursor, true, |_| Ok((vec![], std::collections::HashMap::new(),),),).unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;

        let mut vertices = Vec::with_capacity(vertex_count,);
        for i in 0..vertex_count {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let u = coords[i * 2];
            let v = coords[i * 2 + 1];

            let vertex = Vertex {
                pos:    [x, y, z,],
                // color:  [1.0, 1.0, 1.0,],
                coords: [u, v,],
            };
            vertices.push(vertex,);
        }

        (vertices, mesh.indices.clone(),)
    }

    fn create_vertex_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        vertices: &[Vertex],
    ) -> Buffer
    {
        Buffer::create_device_local_buffer_with_data::<u32, _,>(
            vk_context,
            command_pool,
            transfer_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        )
    }

    fn create_index_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        indices: &[u32],
    ) -> Buffer
    {
        Buffer::create_device_local_buffer_with_data::<u16, _,>(
            vk_context,
            command_pool,
            transfer_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            indices,
        )
    }

    /// Create a descriptor pool to allocate the descriptor sets.
    fn create_descriptor_pool(device: &Device, size: u32,) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty:               vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty:               vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size,];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes,)
            .max_sets(size,)
            .build();

        unsafe { device.create_descriptor_pool(&pool_info, None,).unwrap() }
    }

    /// Create one descriptor set for each uniform buffer.
    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        texture: Texture,
    ) -> Vec<vk::DescriptorSet,>
    {
        let layouts = (0..uniform_buffers.len()).map(|_| layout,).collect::<Vec<_,>>();
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool,)
            .set_layouts(&layouts,)
            .build();
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info,).unwrap() };

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter(),)
            .for_each(|(set, buffer,)| {
                let buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(*buffer,)
                    .offset(0,)
                    .range(size_of::<CameraUBO,>() as vk::DeviceSize,)
                    .build();
                let buffer_infos = [buffer_info,];

                let image_info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,)
                    .image_view(texture.view,)
                    .sampler(texture.sampler.unwrap(),)
                    .build();
                let image_infos = [image_info,];

                let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set,)
                    .dst_binding(0,)
                    .dst_array_element(0,)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER,)
                    .buffer_info(&buffer_infos,)
                    .build();
                let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set,)
                    .dst_binding(1,)
                    .dst_array_element(0,)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER,)
                    .image_info(&image_infos,)
                    .build();

                let descriptor_writes = [ubo_descriptor_write, sampler_descriptor_write,];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[],) }
            },);

        descriptor_sets
    }

    fn create_uniform_buffers(vk_context: &VkContext, count: usize,) -> Vec<Buffer,> {
        let size = size_of::<CameraUBO,>() as vk::DeviceSize;
        let mut buffers = Vec::new();

        for _ in 0..count {
            let buffer = Buffer::create_buffer(
                vk_context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer,);
        }

        buffers
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    pub fn rebuild(
        &mut self,
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    )
    {
        log::debug!("Recreating swapchain.");

        wait_gpu_idle(vk_context,);

        let device = vk_context.device();

        self.swapchain.cleanup(&device,);

        let dimensions = [
            self.swapchain.properties.extent.width,
            self.swapchain.properties.extent.height,
        ];
        let swapchainWrapper = SwapchainWrapper::create_swapchain_and_images(
            &vk_context,
            vk_context.queue_families_indices,
            dimensions,
            command_pool,
            transition_queue,
            msaa_samples,
            depth_format,
        );

        let render_pass = Self::create_render_pass(device, swapchainWrapper.properties, msaa_samples, depth_format,);

        let (pipeline, layout,) = Self::create_pipeline(
            device,
            swapchainWrapper.properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
            self.vertex_binding_descs,
            self.vertex_attribute_descs,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &self.framebuffers,
            render_pass,
            swapchainWrapper.properties,
            self.vertex_buffer,
            self.index_buffer,
            self.model_index_count,
            layout,
            &self.descriptor_sets,
            pipeline,
        );

        let framebuffers = swapchainWrapper
            .images
            .iter()
            .map(|image| {
                let attachments = [
                    swapchainWrapper.color_texture.image.view.unwrap(),
                    swapchainWrapper.depth_texture.image.view.unwrap(),
                    image.view.unwrap(),
                ];

                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass,)
                    .attachments(&attachments,)
                    .width(swapchainWrapper.properties.extent.width,)
                    .height(swapchainWrapper.properties.extent.height,)
                    .layers(1,)
                    .build();
                unsafe { device.create_framebuffer(&framebuffer_info, None,).unwrap() }
            },)
            .collect::<Vec<_,>>();

        self.swapchain = swapchainWrapper;
        // self.swapchain_image_views = swapchain_image_views;
        // self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_layout = layout;
        self.framebuffers = framebuffers;
        self.command_buffers = command_buffers;
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass
    {
        let color_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format,)
            .samples(msaa_samples,)
            .load_op(vk::AttachmentLoadOp::CLEAR,)
            .store_op(vk::AttachmentStoreOp::STORE,)
            .initial_layout(vk::ImageLayout::UNDEFINED,)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,)
            .build();
        let depth_attachement_desc = vk::AttachmentDescription::builder()
            .format(depth_format,)
            .samples(msaa_samples,)
            .load_op(vk::AttachmentLoadOp::CLEAR,)
            .store_op(vk::AttachmentStoreOp::DONT_CARE,)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE,)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE,)
            .initial_layout(vk::ImageLayout::UNDEFINED,)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,)
            .build();
        let resolve_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format,)
            .samples(vk::SampleCountFlags::TYPE_1,)
            .load_op(vk::AttachmentLoadOp::DONT_CARE,)
            .store_op(vk::AttachmentStoreOp::STORE,)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE,)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE,)
            .initial_layout(vk::ImageLayout::UNDEFINED,)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR,)
            .build();
        let attachment_descs = [color_attachment_desc, depth_attachement_desc, resolve_attachment_desc,];

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0,)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,)
            .build();
        let color_attachment_refs = [color_attachment_ref,];

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1,)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,)
            .build();

        let resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2,)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,)
            .build();
        let resolve_attachment_refs = [resolve_attachment_ref,];

        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS,)
            .color_attachments(&color_attachment_refs,)
            .resolve_attachments(&resolve_attachment_refs,)
            .depth_stencil_attachment(&depth_attachment_ref,)
            .build();
        let subpass_descs = [subpass_desc,];

        let subpass_dep = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL,)
            .dst_subpass(0,)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,)
            .src_access_mask(vk::AccessFlags::empty(),)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,)
            .build();
        let subpass_deps = [subpass_dep,];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs,)
            .subpasses(&subpass_descs,)
            .dependencies(&subpass_deps,)
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None,).unwrap() }
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        vertex_binding_descs: vk::VertexInputBindingDescription,
        vertex_attribute_descs: &[vk::VertexInputAttributeDescription],
    ) -> (vk::Pipeline, vk::PipelineLayout,)
    {
        let vertex_source = read_shader_from_file("shaders/shader.vert.spv",);
        let fragment_source = read_shader_from_file("shaders/shader.frag.spv",);

        log::debug!("Compiling shaders...");

        let vertex_shader_module = create_shader_module(device, &vertex_source,);
        let fragment_shader_module = create_shader_module(device, &fragment_source,);

        let entry_point_name = CString::new("main",).unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX,)
            .module(vertex_shader_module,)
            .name(&entry_point_name,)
            .build();
        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT,)
            .module(fragment_shader_module,)
            .name(&entry_point_name,)
            .build();
        let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info,];

        // let vertex_binding_descs = [Vertex::get_binding_description(),];
        // let vertex_attribute_descs = Vertex::get_attribute_descriptions();
        let vertex_binding_descs = [vertex_binding_descs,];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descs,)
            .vertex_attribute_descriptions(&vertex_attribute_descs,)
            .build();

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST,)
            .primitive_restart_enable(false,)
            .build();

        let viewport = vk::Viewport {
            x:         0.0,
            y:         0.0,
            width:     swapchain_properties.extent.width as _,
            height:    swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport,];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: 0, y: 0,
            },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor,];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports,)
            .scissors(&scissors,)
            .build();

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false,)
            .rasterizer_discard_enable(false,)
            .polygon_mode(vk::PolygonMode::FILL,)
            .line_width(1.0,)
            .cull_mode(vk::CullModeFlags::BACK,)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE,)
            .depth_bias_enable(false,)
            .depth_bias_constant_factor(0.0,)
            .depth_bias_clamp(0.0,)
            .depth_bias_slope_factor(0.0,)
            .build();

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .min_sample_shading(1.0)
            // .sample_mask() // null
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true,)
            .depth_write_enable(true,)
            .depth_compare_op(vk::CompareOp::LESS,)
            .depth_bounds_test_enable(false,)
            .min_depth_bounds(0.0,)
            .max_depth_bounds(1.0,)
            .stencil_test_enable(false,)
            .front(Default::default(),)
            .back(Default::default(),)
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all(),)
            .blend_enable(false,)
            .src_color_blend_factor(vk::BlendFactor::ONE,)
            .dst_color_blend_factor(vk::BlendFactor::ZERO,)
            .color_blend_op(vk::BlendOp::ADD,)
            .src_alpha_blend_factor(vk::BlendFactor::ONE,)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO,)
            .alpha_blend_op(vk::BlendOp::ADD,)
            .build();
        let color_blend_attachments = [color_blend_attachment,];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false,)
            .logic_op(vk::LogicOp::COPY,)
            .attachments(&color_blend_attachments,)
            .blend_constants([0.0, 0.0, 0.0, 0.0,],)
            .build();

        let layout = {
            let layouts = [descriptor_set_layout,];
            let push_constant_ranges = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset:      0,
                size:        size_of::<Matrix4<f32,>,>() as _,
            };
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts,)
                .push_constant_ranges(&[push_constant_ranges,],)
                .build();

            unsafe { device.create_pipeline_layout(&layout_info, None,).unwrap() }
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_states_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            // .dynamic_state() null since don't have any dynamic states
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)
            // .base_pipeline_handle() null since it is not derived from another
            // .base_pipeline_index(-1) same
            .build();
        let pipeline_infos = [pipeline_info,];

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None,)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None,);
            device.destroy_shader_module(fragment_shader_module, None,);
        };

        (pipeline, layout,)
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer,>
    {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool,)
            .level(vk::CommandBufferLevel::PRIMARY,)
            .command_buffer_count(framebuffers.len() as _,)
            .build();

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info,).unwrap() };

        buffers.iter().enumerate().for_each(|(i, buffer,)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    // .inheritance_info() null since it's a primary command buffer
                    .build();
                unsafe {
                    device
                        .begin_command_buffer(buffer, &command_buffer_begin_info,)
                        .unwrap()
                };
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0,],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth:   1.0,
                            stencil: 0,
                        },
                    },
                ];
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass,)
                    .framebuffer(framebuffer,)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D {
                            x: 0, y: 0,
                        },
                        extent: swapchain_properties.extent,
                    },)
                    .clear_values(&clear_values,)
                    .build();

                unsafe { device.cmd_begin_render_pass(buffer, &render_pass_begin_info, vk::SubpassContents::INLINE,) };
            }

            // Bind pipeline
            unsafe { device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline,) };

            // Bind vertex buffer
            let vertex_buffers = [vertex_buffer,];
            let offsets = [0,];
            unsafe { device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &offsets,) };

            // Bind index buffer
            unsafe { device.cmd_bind_index_buffer(buffer, index_buffer, 0, vk::IndexType::UINT32,) };

            // Bind descriptor set
            unsafe {
                let null = [];
                device.cmd_bind_descriptor_sets(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[i..=i],
                    &null,
                )
            };

            // Render objects
            let base_rot = Matrix4::from_angle_x(Deg(270.0,),);
            let transform_0 = Matrix4::from_translation(vec3(0.1, 0.0, -1.0,),) * base_rot;
            let transform_1 = Matrix4::from_translation(vec3(-0.1, 0.0, 1.0,),) * base_rot;

            Self::cmd_draw_chalet(device, buffer, index_count, pipeline_layout, transform_0,);

            Self::cmd_draw_chalet(device, buffer, index_count, pipeline_layout, transform_1,);

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer,) };

            // End command buffer
            unsafe { device.end_command_buffer(buffer,).unwrap() };
        },);

        buffers
    }

    /// Pick the first suitable physical device.
    ///
    /// # Requirements
    /// - At least one queue family with one queue supportting graphics.
    /// - At least one queue family with one queue supporting presentation to `surface_khr`.
    /// - Swapchain extension support.
    ///
    /// # Returns
    ///
    /// A tuple containing the physical device and the queue families indices.
    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices,)
    {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device,),)
            .expect("No suitable physical device.",);

        let props = unsafe { instance.get_physical_device_properties(device,) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr(),)
        });

        let (graphics, present,) = Self::find_queue_families(instance, surface, surface_khr, device,);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index:  present.unwrap(),
        };

        (device, queue_families_indices,)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool
    {
        let (graphics, present,) = Self::find_queue_families(instance, surface, surface_khr, device,);
        let extention_support = Self::check_device_extension_support(instance, device,);
        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr,);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device,) };
        graphics.is_some() &&
            present.is_some() &&
            extention_support &&
            is_swapchain_adequate &&
            features.sampler_anisotropy == vk::TRUE
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32,>, Option<u32,>,)
    {
        let mut graphics = None;
        let mut present = None;
        let mut compute = None;
        let mut transfer = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device,) };
        for (index, family,) in props.iter().filter(|f| f.queue_count > 0,).enumerate() {
            let index = index as u32;

            print!("Queue {} with queue a count of {} has: ", index, family.queue_count);

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS,) && graphics.is_none() {
                print!("graphics, ");
                graphics = Some(index,);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr,)
                    .unwrap()
            };
            if present_support && present.is_none() {
                print!("present, ");
                present = Some(index,);
            }

            if family.queue_flags.contains(vk::QueueFlags::COMPUTE,) && compute.is_none() {
                print!("compute, ");
                compute = Some(index,);
            }

            if family.queue_flags.contains(vk::QueueFlags::TRANSFER,) && compute.is_none() {
                print!("transfer, ");
                transfer = Some(index,);
            }

            println!("");

            // if graphics.is_some() && present.is_some() {
            //     break;
            // }
        }

        (graphics, present,)
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice,) -> bool {
        let required_extentions = Self::get_required_device_extensions();

        let extension_props = unsafe { instance.enumerate_device_extension_properties(device,).unwrap() };

        for required in required_extentions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr(),) };
                required == &name
            },);

            if !found {
                return false;
            }
        }

        true
    }

    /// Create the logical device to interact with `device`, a graphics queue
    /// and a presentation queue.
    ///
    /// # Returns
    ///
    /// Return a tuple containing the logical device, the graphics queue and the presentation queue.
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, vk::Queue, vk::Queue,)
    {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities = [1.0f32,];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index,)
                        .queue_priorities(&queue_priorities,)
                        .build()
                },)
                .collect::<Vec<_,>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions.iter().map(|ext| ext.as_ptr(),).collect::<Vec<_,>>();

        let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true,).build();

        let (_layer_names, layer_names_ptrs,) = get_layer_names_and_pointers();

        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos,)
            .enabled_extension_names(&device_extensions_ptrs,)
            .enabled_features(&device_features,);
        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder = device_create_info_builder.enabled_layer_names(&layer_names_ptrs,)
        }
        let device_create_info = device_create_info_builder.build();

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None,)
                .expect("Failed to create logical device.",)
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0,) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0,) };

        (device, graphics_queue, present_queue,)
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name(),]
    }

    fn cmd_draw_chalet(
        device: &Device,
        buffer: vk::CommandBuffer,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        transform: Matrix4<f32,>,
    )
    {
        // Push constants
        unsafe {
            device.cmd_push_constants(
                buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                any_as_u8_slice(&transform,),
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(buffer, index_count as _, 1, 0, 0, 0,) };
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup(&mut self, device: &Device,) {
        unsafe {
            // self.depth_texture.destroy(device,);
            // self.color_texture.destroy(device,);
            self.swapchain.cleanup(device,);
            self.framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None,),);
            device.free_command_buffers(self.command_pool, &self.command_buffers,);
            device.destroy_pipeline(self.pipeline, None,);
            device.destroy_pipeline_layout(self.pipeline_layout, None,);
            device.destroy_render_pass(self.render_pass, None,);
            // self.images
            //     .iter()
            //     .for_each(|v| device.destroy_image_view(v.view.unwrap(), None,),);
            // self.swapchain.destroy_swapchain(self.swapchain_khr, None,);
        }
    }
}
