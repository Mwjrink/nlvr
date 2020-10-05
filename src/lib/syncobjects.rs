use ash::{version::DeviceV1_0, vk, Device};

#[derive(Clone, Copy)]
pub struct SyncObjects {
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub fence:                     vk::Fence,
}

impl SyncObjects {
    pub fn destroy(&self, device: &Device,) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None,);
            device.destroy_semaphore(self.render_finished_semaphore, None,);
            device.destroy_fence(self.fence, None,);
        }
    }
}
