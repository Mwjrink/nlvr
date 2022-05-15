use super::context::*;
use super::utils::*;
use ash::{
    // version::DeviceV1_0, 
    vk, Device};
use std::mem::{align_of, size_of};
use vk_mem::{VirtualAllocation};

pub struct BuffPtr {
    pub handle: VirtualAllocation,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /* VkDevice device;
     * VkBuffer buffer = VK_NULL_HANDLE;
     * VkDeviceMemory memory = VK_NULL_HANDLE;
     * VkDescriptorBufferInfo descriptor;
     * VkDeviceSize size = 0;
     * VkDeviceSize alignment = 0;
     * void* mapped = nullptr;
     * @brief Usage flags to be filled by external source at buffer creation (to query at some later point)
     * VkBufferUsageFlags usageFlags;
     * @brief Memory property flags to be filled by external source at buffer creation (to query at some later
     * point)
     * VkMemoryPropertyFlags memoryPropertyFlags;
     * VkResult map(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
     * void unmap();
     * VkResult bind(VkDeviceSize offset = 0);
     * void setupDescriptor(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
     * void copyTo(void* data, VkDeviceSize size);
     * VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
     * VkResult invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
     * void destroy();
     */
}

impl Buffer {
    /// Create a buffer and allocate its memory.
    ///
    /// # Returns
    ///
    /// The buffer, its memory and the actual size in bytes of the
    /// allocated memory since in may differ from the requested size.
    pub fn create_buffer(
        vk_context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        // println!("Device Alloc: {}", size);

        let device = vk_context.device();
        let buffer = {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();
            unsafe { device.create_buffer(&buffer_info, None).unwrap() }
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type = Self::find_memory_type(mem_requirements, vk_context.get_mem_properties(), mem_properties);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type)
                .build();
            unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
        };

        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };

        Buffer {
            buffer,
            memory,
            size: mem_requirements.size,
        }
    }

    /// Find a memory type in `mem_properties` that is suitable
    /// for `requirements` and supports `required_properties`.
    ///
    /// # Returns
    ///
    /// The index of the memory type from `mem_properties`.
    pub fn find_memory_type(
        requirements: vk::MemoryRequirements,
        mem_properties: vk::PhysicalDeviceMemoryProperties,
        required_properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        for i in 0..mem_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type.")
    }

    /// Copy the `size` first bytes of `src` into `dst`.
    ///
    /// It's done using a command buffer allocated from
    /// `command_pool`. The command buffer is cubmitted tp
    /// `transfer_queue`.
    pub fn copy_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
        dst_offset: u64,
    ) {
        execute_one_time_commands(&device, command_pool, transfer_queue, |buffer| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset,
                size,
            };
            let regions = [region];

            unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) };
        });
    }

    /// Create a buffer and it's gpu  memory and fill it.
    ///
    /// This function internally creates an host visible staging buffer and
    /// a device local buffer. The data is first copied from the cpu to the
    /// staging buffer. Then we copy the data from the staging buffer to the
    /// final buffer using a one-time command buffer.
    pub fn create_device_local_buffer_with_data<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Buffer {
        let device = vk_context.device();
        let size = (data.len() * size_of::<T>()) as vk::DeviceSize;
        let Buffer {
            buffer: staging_buffer,
            memory: staging_memory,
            size: staging_mem_size,
        } = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(data);
            device.unmap_memory(staging_memory);
        };

        let buffer = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer,
            buffer.buffer,
            buffer.size,
            0,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        };

        buffer
    }

    pub fn create_device_local_buffer(
        vk_context: &VkContext,
        usage: vk::BufferUsageFlags,
        size: vk::DeviceSize,
    ) -> Buffer {
        let buffer = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        buffer
    }

    pub fn transfer_to_device_local_buffer<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        buffer: &Buffer,
        data: &[T],
        buffer_offset: u64,
    ) -> vk::DeviceSize {
        let device = vk_context.device();
        let size = (data.len() * size_of::<T>()) as vk::DeviceSize;
        // println!("Transfer to device local size: {}", size);
        let Buffer {
            buffer: staging_buffer,
            memory: staging_memory,
            size: staging_mem_size,
        } = Self::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(data);
            device.unmap_memory(staging_memory);
        };

        Self::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer,
            buffer.buffer,
            size,
            buffer_offset,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        };

        size
    }

    pub fn cleanup(&mut self, vk_context: &VkContext) {
        log::debug!("Dropping buffer.");

        let device = vk_context.device();
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}
