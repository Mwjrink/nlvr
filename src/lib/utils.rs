// utility functions
use ash::{version::DeviceV1_0, vk, Device};

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}

/// Create a one time use command buffer and pass it to `executor`.
pub fn execute_one_time_commands<F: FnOnce(vk::CommandBuffer)>(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    executor: F,
) {
    let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();

        unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] }
    };
    let command_buffers = [command_buffer];

    // Begin recording
    {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info).unwrap() };
    }

    // Execute user function
    executor(command_buffer);

    // End recording
    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    // Submit and wait
    {
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers).build();
        let submit_infos = [submit_info];
        unsafe {
            device.queue_submit(queue, &submit_infos, vk::Fence::null()).unwrap();
            device.queue_wait_idle(queue).unwrap();
        };
    }

    // Free
    unsafe { device.free_command_buffers(command_pool, &command_buffers) };
}
