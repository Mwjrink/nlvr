use ash::vk;

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format:       vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent:       vk::Extent2D,
}
