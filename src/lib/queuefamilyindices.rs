#[derive(Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index:  u32,
    pub transfer_index: u32,
}
