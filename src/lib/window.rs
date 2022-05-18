use super::surface::OutputSurface;
use ash::{vk, Entry, Instance};
use raw_window_handle::HasRawWindowHandle;

pub struct Window {
    window_handle: Box<dyn HasRawWindowHandle>,
    dimensions: [u32; 2],
}

impl Window {
    pub fn create(window_handle: Box<dyn HasRawWindowHandle>, dimensions: [u32; 2]) -> Self {
        Self {
            window_handle,
            dimensions,
        }
    }
}

impl OutputSurface for Window {
    fn get_dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    fn create_surface(&'_ self, instance: &Instance, entry: &Entry) -> vk::SurfaceKHR {
        unsafe { ash_window::create_surface(&entry, &instance, self.window_handle.as_ref(), None).unwrap() }
    }

    fn get_required_extensions(&'_ self) -> Vec<&std::ffi::CStr> {
        ash_window::enumerate_required_extensions(self.window_handle.as_ref()).unwrap()
    }
}
