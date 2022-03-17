extern crate log;
// extern crate vk_mem;

pub mod lib {
    mod buffer;
    mod context;
    mod debug;
    mod fs;
    mod image;
    mod inflightframedata;
    mod queuefamilyindices;
    pub mod renderable;
    pub mod renderinstance;
    mod shaders;
    mod swapchain;
    mod swapchainproperties;
    mod swapchainsupportdetails;
    mod syncobjects;
    mod texture;
    pub mod ubo;
    mod utils;
    mod vertex;
    mod material;
    mod mesh;
}

/*
TODO
- multiple pipelines
    - will allow for menus and ui
    - pass in vertex and fragment shader paths
- figure out the renderable return object or handle situation



*/