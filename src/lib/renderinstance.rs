use super::{
    buffer::*,
    context::*,
    debug::*,
    fs,
    inflightframedata::*,
    queuefamilyindices::*,
    renderable::*,
    shaders::*,
    swapchain::*,
    swapchainproperties::*,
    swapchainsupportdetails::*,
    syncobjects::*,
    texture::*,
    ubo::*,
    utils::*,
    vertex::*, //vulkan::*,
};
use ash::{
    extensions::{
        ext::DebugReport,
        khr::{Surface, Swapchain},
    },
    util::Align,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{self, CommandBuffer},
    Device, Entry, Instance,
};
use cgmath::{vec3, Deg, Matrix4};
use galloc::{Galloc, MemoryBlock};
use std::{
    ffi::{CStr, CString},
    mem::{align_of, size_of},
};
use winit::window::Window;

const MAX_FRAMES_IN_FLIGHT: u32 = 3;
const PAGE_SIZE: u64 = 2147483648; // 4294967296

pub struct RenderInstance<T: UBO + Copy> {
    vk_context: VkContext,
    //
    pub swapchain: SwapchainWrapper,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    msaa_samples: vk::SampleCountFlags,
    depth_format: vk::Format,
    descriptor_set_layout: vk::DescriptorSetLayout,
    command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
    in_flight_frames: InFlightFrames,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // this is here because it depends on the renderpass
    framebuffers: Vec<vk::Framebuffer>,
    // TODO: this will be known at compile time when this is all finalized so make this an array eventually
    // TODO: maybe make this a generic parameter
    vertex_attribute_descs: Vec<vk::VertexInputAttributeDescription>,
    vertex_binding_descs: vk::VertexInputBindingDescription,

    // make these two Option<vk::Queue> and then use them if they exist but fall back on graphics queue
    transfer_queue: vk::Queue,

    // this is the master pool, the command buffers are run on this and they consist of many models, vertices, indices
    // and texture index for a texture array
    // model_index_count:      usize,
    vertex_buffer: Buffer,
    vertex_alloc: Galloc, // TODO: use VMA when it implements buffer sub allocation

    index_buffer: Buffer,
    index_alloc: Galloc, // TODO: use VMA when it implements buffer sub allocation

    renderables: Vec<Renderable>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_buffers: Vec<CommandBuffer>, // TODO should this be here?

    // TODO should this go here?
    global_uniform_buffers: Vec<Buffer>,

    uniform_buffers_align: [Align<T>; MAX_FRAMES_IN_FLIGHT as usize],

    ubo_data: Option<T>,
}

impl<T: UBO + Copy> RenderInstance<T> {
    pub fn create(window: &Window) -> RenderInstance<T> {
        log::debug!("Creating application.");

        let frames_in_flight = 3;

        let entry = Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry, window);

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let (physical_device, queue_families_indices) = Self::pick_physical_device(&instance, &surface, surface_khr);

        let test = unsafe { instance.get_physical_device_properties(physical_device) };

        println!("test: {:?}", test);

        // move to a queue class potentially
        let (device, graphics_queue, present_queue, transfer_queue) =
            Self::create_logical_device_with_graphics_queue(&instance, physical_device, queue_families_indices);

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

        let command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::empty(),
        );

        let depth_format = Self::find_depth_format(&vk_context);
        let msaa_samples = vk_context.get_max_usable_sample_count();

        let swapchain = SwapchainWrapper::create_swapchain_and_images(
            &vk_context,
            queue_families_indices,
            [window.inner_size().width, window.inner_size().height],
            command_pool,
            graphics_queue,
            msaa_samples,
            depth_format,
        );

        let render_pass =
            Self::create_render_pass(vk_context.device(), swapchain.properties, msaa_samples, depth_format);
        let descriptor_set_layout =
            Self::create_descriptor_set_layout(vk_context.device(), T::get_descriptor_set_layout_binding());
        let (pipeline, layout) = Self::create_pipeline(
            vk_context.device(),
            swapchain.properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
            Vertex::get_binding_description(),
            &Vertex::get_attribute_descriptions(),
        );

        let transient_command_pool = Self::create_command_pool(
            vk_context.device(),
            queue_families_indices,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain
                .images
                .iter()
                .map(|image| image.view.unwrap())
                .collect::<Vec<vk::ImageView>>(),
            swapchain.color_texture,
            swapchain.depth_texture,
            render_pass,
            swapchain.properties,
        );

        log::debug!("Create sync objects");

        let in_flight_frames = Self::create_sync_objects(vk_context.device());

        let global_uniform_buffers = Self::create_uniform_buffers(&vk_context, MAX_FRAMES_IN_FLIGHT as _);

        let descriptor_pool = Self::create_descriptor_pool(vk_context.device(), MAX_FRAMES_IN_FLIGHT as _);
        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &global_uniform_buffers
                .iter()
                .map(|buff| buff.buffer)
                .collect::<Vec<vk::Buffer>>(),
            // texture, // global texture atlas
        );

        let size = size_of::<T>() as vk::DeviceSize;
        let uniform_buffers_align = unsafe {
            [
                {
                    let data_ptr = vk_context
                        .device()
                        .map_memory(global_uniform_buffers[0].memory, 0, size, vk::MemoryMapFlags::empty())
                        .unwrap();
                    ash::util::Align::new(data_ptr, align_of::<f32>() as _, size)
                },
                {
                    let data_ptr = vk_context
                        .device()
                        .map_memory(global_uniform_buffers[1].memory, 0, size, vk::MemoryMapFlags::empty())
                        .unwrap();
                    ash::util::Align::new(data_ptr, align_of::<f32>() as _, size)
                },
                {
                    let data_ptr = vk_context
                        .device()
                        .map_memory(global_uniform_buffers[2].memory, 0, size, vk::MemoryMapFlags::empty())
                        .unwrap();
                    ash::util::Align::new(data_ptr, align_of::<f32>() as _, size)
                },
            ]
        };

        Self {
            vk_context,
            pipeline_layout: layout,
            pipeline,
            msaa_samples,
            descriptor_set_layout,
            swapchain,
            render_pass,
            command_pool,
            transient_command_pool,
            in_flight_frames,
            framebuffers: swapchain_framebuffers,
            vertex_binding_descs: Vertex::get_binding_description(),
            vertex_attribute_descs: Vec::from(Vertex::get_attribute_descriptions()),
            graphics_queue,
            present_queue,
            transfer_queue,
            depth_format,
            vertex_buffer: Buffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            // TODO cull 0 size blocks in Galloc tree
            vertex_alloc: Galloc::new(0, None),
            index_buffer: Buffer {
                buffer: vk::Buffer::null(),
                memory: vk::DeviceMemory::null(),
                size: 0,
            },
            // TODO cull 0 size blocks in Galloc tree
            index_alloc: Galloc::new(0, None),
            renderables: Vec::new(),

            descriptor_pool,
            descriptor_sets,

            command_buffers: Vec::new(),

            // camera: Default::default(),
            global_uniform_buffers,

            uniform_buffers_align,

            ubo_data: None,
        }
    }

    // TODO
    /*
    the model assets should hold a pointer to the buffer with vertex/index data
    the material assets should hold pointers to the pipelines and descriptors
    In the code where you load assets (models and materials), the model assets should hold a pointer to the buffer with
    vertex/index data, and the material assets should hold pointers to the pipelines and descriptors. Generally, you want
    to separate material data in to "per-shader" data (i.e. a set of pipelines for a certain material type, for example
    pipelines that renders deferred geometry) as well as the per-instance material data (which holds the material
    attributes and reside inside descriptor sets for uniforms). Material assets should point to which shader it
    should use, and contain its own set of attribute/uniform data.

    Advanced usecases will have cached and prebuilt pipelines that is loaded from disk, as well as more sofisticated
    systems for descriptor sets/pools.

    This also ties in to the "per-shader" data mentioned above; A PBS shader will hold the per-shader data (the pipelines
    used to render into some renderpass), the material instance will hold a reference to which shader it should use, as
    well as its own set of attribute/uniform data inside descriptor sets.
    */

    // TODO texture should not be optional, in theory
    pub fn renderable_from_file(&mut self, model_path: String, texture_path: Option<String>) -> Renderable {
        // if let Some(tex_path) = texture_path {
        //     let texture =
        //         Texture::create_texture_image(&self.vk_context, self.command_pool, self.graphics_queue, tex_path);
        // }

        let (vertices, indices) = Self::load_model(model_path.clone());

        // vertex
        let vertex_buffer_ptr;
        {
            if self.vertex_buffer.size == 0 {
                // add to overall vertex buffers and overall index count/buffers?
                let vertex_buffer = Self::create_vertex_buffer(&self.vk_context);

                self.vertex_buffer = vertex_buffer;
                self.vertex_alloc.free(MemoryBlock {
                    offset: 0,
                    size: PAGE_SIZE as usize,
                });
            }

            // TODO no unwrap here, handle running out of memory
            let vertex_block = self
                .vertex_alloc
                .allocate(vertices.len() * size_of::<Vertex>())
                .unwrap();
            let vertices_size = Self::transfer_vertices(
                &self.vk_context,
                self.transient_command_pool,
                self.transfer_queue,
                &self.vertex_buffer,
                &vertices,
                vertex_block.offset as u64,
            );
            vertex_buffer_ptr = BuffPtr {
                offset: vertex_block.offset as u64,
                size: vertices_size,
            };
        }

        // index
        let index_buffer_ptr;
        {
            if self.index_buffer.size == 0 {
                // add to overall vertex buffers and overall index count/buffers?
                let index_buffer = Self::create_index_buffer(&self.vk_context);

                self.index_buffer = index_buffer;
                self.index_alloc.free(MemoryBlock {
                    offset: 0,
                    size: PAGE_SIZE as usize,
                });
            }

            // TODO manually offset indices before upload based on offset of vertex_ptr
            // This might not be necessary with the given draw command?

            // TODO no unwrap here, handle running out of memory
            let index_block = self.index_alloc.allocate(indices.len() * size_of::<u32>()).unwrap();
            let indices_size = Self::transfer_indices(
                &self.vk_context,
                self.transient_command_pool,
                self.transfer_queue,
                &self.index_buffer,
                &indices,
                index_block.offset as u64,
            );
            index_buffer_ptr = BuffPtr {
                offset: index_block.offset as u64,
                size: indices_size,
            };
        }

        // get some ptrs to the vertex and index buffers

        // model_index_count:      usize,
        // vertex_buffer:          Buffer,
        // vertex_size:            usize,
        // index_buffer:           Buffer,
        // index_size:             usize,

        // self.vertex_buffer.add some shit to it

        Renderable {
            model_index_count: indices.len(),
            // texture,
            // descriptor_pool,
            // descriptor_sets,
            // uniform_buffers,
            vertex_buffer_ptr,
            index_buffer_ptr,
            //
            asset_path: model_path,
            // command_buffers,
            instances: Vec::new(),
        }
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            sync_objects_vec.push(SyncObjects::create(device))
        }

        InFlightFrames::new(sync_objects_vec)
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 2, 148))
            .build();

        let extension_names = ash_window::enumerate_required_extensions(window).unwrap();

        let mut extension_names = extension_names.iter().map(|ext| ext.as_ptr()).collect::<Vec<_>>();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugReport::name().as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    fn load_model(asset_path: String) -> (Vec<Vertex>, Vec<u32>) {
        let mut cursor = fs::load(asset_path);
        let (models, _) =
            tobj::load_obj_buf(&mut cursor, true, |_| Ok((vec![], std::collections::HashMap::new()))).unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let u = coords[i * 2];
            let v = coords[i * 2 + 1];

            let vertex = Vertex {
                pos: [x, y, z],
                // color:  [1.0, 1.0, 1.0,],
                coords: [u, v],
            };
            vertices.push(vertex);
        }

        (vertices, mesh.indices.clone())
    }

    fn create_vertex_buffer(vk_context: &VkContext) -> Buffer {
        Buffer::create_device_local_buffer(vk_context, vk::BufferUsageFlags::VERTEX_BUFFER, PAGE_SIZE)
    }

    fn create_index_buffer(vk_context: &VkContext) -> Buffer {
        Buffer::create_device_local_buffer(vk_context, vk::BufferUsageFlags::INDEX_BUFFER, PAGE_SIZE)
    }

    fn transfer_vertices(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        vertex_buffer: &Buffer,
        vertices: &[Vertex],
        offset: u64,
    ) -> u64 {
        Buffer::transfer_to_device_local_buffer::<u32, _>(
            vk_context,
            command_pool,
            transfer_queue,
            vertex_buffer,
            vertices,
            offset,
        )
    }

    fn transfer_indices(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        index_buffer: &Buffer,
        indices: &[u32],
        offset: u64,
    ) -> u64 {
        Buffer::transfer_to_device_local_buffer::<u16, _>(
            vk_context,
            command_pool,
            transfer_queue,
            index_buffer,
            indices,
            offset,
        )
    }

    /// Create a descriptor pool to allocate the descriptor sets.
    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(size)
            .build();

        unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() }
    }

    /// Create one descriptor set for each uniform buffer.
    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        // texture: Texture, // global texture atlas
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len()).map(|_| layout).collect::<Vec<_>>();
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .build();
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, buffer)| {
                let buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(*buffer)
                    .offset(0)
                    .range(size_of::<T>() as vk::DeviceSize)
                    .build();
                let buffer_infos = [buffer_info];

                // let image_info = vk::DescriptorImageInfo::builder()
                //     .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                //     .image_view(texture.image.view.unwrap())
                //     .sampler(texture.sampler.unwrap())
                //     .build();
                // let image_infos = [image_info];

                let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos)
                    .build();
                // let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                //     .dst_set(*set)
                //     .dst_binding(1)
                //     .dst_array_element(0)
                //     .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                //     .image_info(&image_infos)
                //     .build();

                let descriptor_writes = [ubo_descriptor_write]; //, sampler_descriptor_write];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }
            });

        descriptor_sets
    }

    fn create_uniform_buffers(vk_context: &VkContext, count: usize) -> Vec<Buffer> {
        let size = size_of::<T>() as vk::DeviceSize;
        let mut buffers = Vec::new();

        for _ in 0..count {
            let buffer = Buffer::create_buffer(
                vk_context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer);
        }

        buffers
    }

    fn find_depth_format(vk_context: &VkContext) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        vk_context
            .find_supported_format(
                &candidates,
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
            .expect("Failed to find a supported depth format")
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        image_views
            .iter()
            .map(|view| {
                [
                    color_texture.image.view.unwrap(),
                    depth_texture.image.view.unwrap(),
                    *view,
                ]
            })
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    .layers(1)
                    .build();
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    pub fn rebuild(&mut self) {
        log::debug!("Recreating swapchain.");

        self.wait_gpu_idle();

        let device = self.vk_context.device();

        self.swapchain.cleanup(&device);
        self.cleanup(device);

        let dimensions = [
            self.swapchain.properties.extent.width,
            self.swapchain.properties.extent.height,
        ];
        let swapchain_wrapper = SwapchainWrapper::create_swapchain_and_images(
            &self.vk_context,
            self.vk_context.queue_families_indices,
            dimensions,
            self.command_pool,
            self.graphics_queue,
            self.msaa_samples,
            self.depth_format,
        );

        let render_pass = Self::create_render_pass(
            device,
            swapchain_wrapper.properties,
            self.msaa_samples,
            self.depth_format,
        );

        let (pipeline, layout) = Self::create_pipeline(
            device,
            swapchain_wrapper.properties,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
            self.vertex_binding_descs,
            &self.vertex_attribute_descs,
        );

        let framebuffers = swapchain_wrapper
            .images
            .iter()
            .map(|image| {
                let attachments = [
                    swapchain_wrapper.color_texture.image.view.unwrap(),
                    swapchain_wrapper.depth_texture.image.view.unwrap(),
                    image.view.unwrap(),
                ];

                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_wrapper.properties.extent.width)
                    .height(swapchain_wrapper.properties.extent.height)
                    .layers(1)
                    .build();
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        // this needs to be rebuilt in the renderable objects?
        let command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &framebuffers,
            render_pass,
            swapchain_wrapper.properties,
            self.vertex_buffer.buffer,
            self.index_buffer.buffer,
            &self.renderables,
            self.renderables.iter().map(|element| element.instances.len()).sum(),
            layout,
            &self.descriptor_sets,
            pipeline,
        );

        self.swapchain = swapchain_wrapper;
        // self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
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
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let depth_attachement_desc = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();
        let attachment_descs = [color_attachment_desc, depth_attachement_desc, resolve_attachment_desc];

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .resolve_attachments(&resolve_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .build();
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build();
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps)
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        vertex_binding_descs: vk::VertexInputBindingDescription,
        vertex_attribute_descs: &[vk::VertexInputAttributeDescription],
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_source = read_shader_from_file("shaders/shader.vert.spv");
        let fragment_source = read_shader_from_file("shaders/shader.frag.spv");

        log::debug!("Compiling shaders...");

        let vertex_shader_module = create_shader_module(device, &vertex_source);
        let fragment_shader_module = create_shader_module(device, &fragment_source);

        let entry_point_name = CString::new("main").unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name)
            .build();
        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name)
            .build();
        let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

        // let vertex_binding_descs = [Vertex::get_binding_description(),];
        // let vertex_attribute_descs = Vertex::get_attribute_descriptions();
        let vertex_binding_descs = [vertex_binding_descs];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs)
            .build();

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors)
            .build();

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
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
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default())
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();
        let color_blend_attachments = [color_blend_attachment];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let layout = {
            let layouts = [descriptor_set_layout];
            let push_constant_ranges = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: size_of::<Matrix4<f32>>() as _,
            };
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&[push_constant_ranges])
                .build();

            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
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
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        };

        (pipeline, layout)
    }

    fn create_descriptor_set_layout(
        device: &Device,
        ubo_binding: vk::DescriptorSetLayoutBinding,
    ) -> vk::DescriptorSetLayout {
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let bindings = [ubo_binding, sampler_binding];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings).build();

        unsafe { device.create_descriptor_set_layout(&layout_info, None).unwrap() }
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        renderables: &[Renderable],
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _)
            .build();

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    // .inheritance_info() null since it's a primary command buffer
                    .build();
                unsafe { device.begin_command_buffer(buffer, &command_buffer_begin_info).unwrap() };
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                    },
                ];
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
                    })
                    .clear_values(&clear_values)
                    .build();

                unsafe { device.cmd_begin_render_pass(buffer, &render_pass_begin_info, vk::SubpassContents::INLINE) };
            }

            // Bind pipeline
            unsafe { device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, graphics_pipeline) };

            // Bind vertex buffer
            let vertex_buffers = [vertex_buffer];
            // let offsets = renderables
            //     .iter()
            //     .map(|r| r.vertex_buffer_ptr.offset,)
            //     .collect::<Vec<u64,>>();
            unsafe { device.cmd_bind_vertex_buffers(buffer, 0, &vertex_buffers, &[0]) };

            // Bind index buffer
            unsafe { device.cmd_bind_index_buffer(buffer, index_buffer, 0, vk::IndexType::UINT32) };

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
            let base_rot = Matrix4::from_angle_x(Deg(270.0));
            let transform_0 = Matrix4::from_translation(vec3(0.1, 0.0, -1.0)) * base_rot;
            let transform_1 = Matrix4::from_translation(vec3(-0.1, 0.0, 1.0)) * base_rot;

            Self::cmd_draw_chalet(device, buffer, index_count, pipeline_layout, transform_0);

            Self::cmd_draw_chalet(device, buffer, index_count, pipeline_layout, transform_1);

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End command buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
        });

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
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let (graphics, present, transfer) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
            transfer_index: transfer.unwrap(),
        };

        (device, queue_families_indices)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present, transfer) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extention_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_adequate = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extention_support
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE
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
    ) -> (Option<u32>, Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;
        // let mut compute = None;
        let mut transfer = None;
        // compute queue or use graphics?

        // Queue 0 with queue a count of 16 has: graphics, present, compute, transfer, SPARSE_BINDING,
        // Queue 1 with queue a count of 2 has: transfer, SPARSE_BINDING,
        // Queue 2 with queue a count of 8 has: present, compute, transfer, SPARSE_BINDING,

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            print!("Queue {} with queue a count of {} has: ", index, family.queue_count);

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                print!("graphics, ");
                if graphics.is_none() {
                    graphics = Some(index);
                }
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };
            if present_support {
                print!("present, ");
                if present.is_none() {
                    present = Some(index);
                }
            }

            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                print!("compute, ");
                // compute = Some(index,);
            }

            if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                print!("transfer, ");
                if index == 1 {
                    transfer = Some(index);
                }
            }

            if family.queue_flags.contains(vk::QueueFlags::RESERVED_5_KHR) {
                print!("RESERVED_5_KHR, ");
            }

            if family.queue_flags.contains(vk::QueueFlags::RESERVED_6_KHR) {
                print!("RESERVED_6_KHR, ");
            }

            if family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                print!("SPARSE_BINDING, ");
            }

            println!("");

            // if graphics.is_some() && present.is_some() {
            //     break;
            // }
        }

        (graphics, present, transfer)
    }

    fn create_command_pool(
        device: &Device,
        queue_families_indices: QueueFamiliesIndices,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families_indices.graphics_index)
            .flags(create_flags)
            .build();

        unsafe { device.create_command_pool(&command_pool_info, None).unwrap() }
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extentions = Self::get_required_device_extensions();

        let extension_props = unsafe { instance.enumerate_device_extension_properties(device).unwrap() };

        for required in required_extentions.iter() {
            let found = extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            });

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
    ) -> (Device, vk::Queue, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let transfer_family_index = queue_families_indices.transfer_index;
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index, transfer_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions.iter().map(|ext| ext.as_ptr()).collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true).build();

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);
        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder = device_create_info_builder.enabled_layer_names(&layer_names_ptrs)
        }
        let device_create_info = device_create_info_builder.build();

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_family_index, 0) };

        (device, graphics_queue, present_queue, transfer_queue)
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name()]
    }

    fn cmd_draw_chalet(
        device: &Device,
        buffer: vk::CommandBuffer,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        transform: Matrix4<f32>,
    ) {
        // Push constants
        unsafe {
            device.cmd_push_constants(
                buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                any_as_u8_slice(&transform),
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(buffer, index_count as _, 1, 0, 0, 0) };
    }

    // TODO allow any number of ubos to be passed?
    // TODO make the memory for uniform buffers (stored align) a big buffer and reallocate if it is too small here:
    pub fn update_uniform_buffers(&mut self, ubo: T) {
        self.ubo_data = Some(ubo);
    }

    pub fn draw_frame(&mut self) -> bool {
        // take hot param as mut and use its swapchain index
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context
                .device()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = unsafe {
            self.swapchain.swapchain.acquire_next_image(
                self.swapchain.swapchain_khr,
                std::u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.vk_context.device().reset_fences(&wait_fences).unwrap() };

        if let Some(ubo) = self.ubo_data {
            let ubos = [ubo];

            // TODO use offsets?
            // TODO this should be fine cause of memory fences in vulkan and shit but I can just flush to be safe here:
            // device.flush_mapped_memory_ranges
            self.uniform_buffers_align[image_index as usize].copy_from_slice(&ubos);
        }

        let device = self.vk_context.device();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(self.graphics_queue, &submit_infos, in_flight_fence)
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain.swapchain_khr];
        let images_indices = [image_index];

        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices)
                // .results() null since we only have one swapchain
                .build();
            let result = unsafe {
                self.swapchain
                    .swapchain
                    .queue_present(self.present_queue, &present_info)
            };
            match result {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return true;
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }
        }

        false
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup(&self, device: &Device) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| device.destroy_framebuffer(*f, None));
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
        }
    }

    pub fn wait_gpu_idle(&self) {
        unsafe { self.vk_context.device().device_wait_idle().unwrap() };
    }
}

impl<T: UBO + Copy> Drop for RenderInstance<T> {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        let device = self.vk_context.device();

        self.swapchain.cleanup(device);
        self.cleanup(device);
        self.in_flight_frames.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
                device.unmap_memory(self.global_uniform_buffers[i].memory);
                self.global_uniform_buffers[i].cleanup(&self.vk_context);
            }

            self.index_buffer.cleanup(&self.vk_context);
            self.vertex_buffer.cleanup(&self.vk_context);
            // self.texture.destroy(device,);
            device.destroy_command_pool(self.transient_command_pool, None);
            device.destroy_command_pool(self.command_pool, None);

            device.destroy_device(None);
        }
    }
}
