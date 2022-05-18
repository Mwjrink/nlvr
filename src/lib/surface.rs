use ash::{vk, Entry, Instance};
use std::ffi::CStr;

pub trait OutputSurface {
    fn get_dimensions(&self) -> [u32; 2];

    fn create_surface(&self, instance: &Instance, entry: &Entry) -> vk::SurfaceKHR;

    fn get_required_extensions(&self) -> Vec<&CStr>;
}
