use super::surface::OutputSurface;
use ash::{vk, Entry, Instance};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub struct Window {
    window_handle: Box<dyn HasRawWindowHandle>,
    display_handle: Box<dyn HasRawDisplayHandle>,
    dimensions: [u32; 2],
}

impl Window {
    pub fn create(
        window_handle: Box<dyn HasRawWindowHandle>,
        display_handle: Box<dyn HasRawDisplayHandle>,
        dimensions: [u32; 2],
    ) -> Self {
        Self {
            window_handle,
            display_handle,
            dimensions,
        }
    }
}

impl OutputSurface for Window {
    fn get_dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    fn create_surface(&'_ self, instance: &Instance, entry: &Entry) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                self.display_handle.raw_display_handle(),
                self.window_handle.raw_window_handle(),
                None,
            )
            .unwrap()
        }
    }

    fn get_required_extensions(&'_ self) -> Vec<*const i8> {
        ash_window::enumerate_required_extensions(self.display_handle.raw_display_handle()).unwrap().to_vec()
    }
}
