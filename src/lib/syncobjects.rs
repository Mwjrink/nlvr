use ash::{
    // version::DeviceV1_0, 
    vk, Device};

#[derive(Clone, Copy)]
pub struct SyncObjects {
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub fence:                     vk::Fence,
}

impl SyncObjects {
    pub fn create(device: &Device,) -> SyncObjects {
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
            unsafe { device.create_semaphore(&semaphore_info, None,).unwrap() }
        };

        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
            unsafe { device.create_semaphore(&semaphore_info, None,).unwrap() }
        };

        let in_flight_fence = {
            let fence_info = vk::FenceCreateInfo::builder()
                .flags(vk::FenceCreateFlags::SIGNALED,)
                .build();
            unsafe { device.create_fence(&fence_info, None,).unwrap() }
        };

        SyncObjects {
            image_available_semaphore,
            render_finished_semaphore,
            fence: in_flight_fence,
        }
    }

    pub fn destroy(&self, device: &Device,) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None,);
            device.destroy_semaphore(self.render_finished_semaphore, None,);
            device.destroy_fence(self.fence, None,);
        }
    }
}
