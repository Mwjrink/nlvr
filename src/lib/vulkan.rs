use ash::{
    extensions::{
        ext::DebugReport,
        khr::{Surface, Swapchain},
    },
    version::{EntryV1_0, InstanceV1_0},
    vk, Entry, Instance,
};
use std::ffi::{CStr, CString};

use winit::window::Window;

pub mod swapchain;

const ENABLE_VALIDATION_LAYERS: bool = false;
const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

#[derive(Clone, Copy)]
struct QueueFamiliesIndices {
    graphics_index: u32,
    present_index: u32,
}

pub struct VulkanApp {}

impl VulkanApp {
    pub fn new(window: &Window) -> Self {
        log::trace!("Application init");

        let entry = Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry, window);

        let surface = Surface::new(&entry, &instance);
        let surface_khr =
            unsafe { ash_window::create_surface(&entry, &instance, window, None).unwrap() };

        let _debug_report_callback = {
            if ENABLE_VALIDATION_LAYERS {
                let create_info = vk::DebugReportCallbackCreateInfoEXT::builder()
                    .flags(vk::DebugReportFlagsEXT::all())
                    .pfn_callback(Some(super::debug::vulkan_debug_callback))
                    .build();
                let debug_report = DebugReport::new(&entry, &instance);
                let debug_report_callback = unsafe {
                    debug_report
                        .create_debug_report_callback(&create_info, None)
                        .unwrap()
                };
                Some((debug_report, debug_report_callback))
            } else {
                None
            }
        };

        let (_physical_device, _queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr);

        //

        Self {}
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Application Name").unwrap();
        let engine_name = CString::new("Engine Name").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 0, 0))
            .build();

        let extension_names = ash_window::enumerate_required_extensions(window).unwrap();
        let mut extension_names = extension_names
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugReport::name().as_ptr());
        }

        // let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();
        let layer_names = REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).unwrap())
            .collect::<Vec<_>>();

        let layer_names_ptrs = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            for required in REQUIRED_LAYERS.iter() {
                let found = entry
                    .enumerate_instance_layer_properties()
                    .unwrap()
                    .iter()
                    .any(|layer| {
                        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                        let name = name.to_str().expect("Failed to get layer name pointer");
                        required == &name
                    });

                if !found {
                    panic!("Validation layer not supported: {}", required);
                }
            }

            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        let instance_create_info = instance_create_info.build();

        // second param is allocation callbacks
        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
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

        let pdprops = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(pdprops.device_name.as_ptr())
        });

        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_families_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_families_indices)
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extention_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_adequate = {
            let details = swapchain::SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extention_support
            && is_swapchain_adequate
            && features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extentions = [Swapchain::name()];

        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

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
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(device, index, surface_khr)
                    .unwrap()
            };
            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }

    pub fn recreate_swapchain(&self) {}

    pub fn draw_frame(&self) -> bool {
        return false;
    }
}
