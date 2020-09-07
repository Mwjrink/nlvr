use ash::vk;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

pub unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugReportFlagsEXT,
    typ: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    if flag == vk::DebugReportFlagsEXT::DEBUG {
        log::debug!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::INFORMATION {
        log::info!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::PERFORMANCE_WARNING {
        log::warn!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    } else {
        log::error!("{:?} - {:?}", typ, CStr::from_ptr(p_message));
    }
    vk::FALSE
}
